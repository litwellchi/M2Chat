import os
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download
import torch.nn.functional as F

import diffusers
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v):
        q = q.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        k = k.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        v = v.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        attn_output, attn_weights = self.multihead_attn(q, k, v)
        attn_output = attn_output.permute(1, 0, 2)  # [ batch_size, seq_len, hidden_size]
        out = self.linear(attn_output)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size=4096):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.ffn(x)
    

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.crossAttention = CrossAttention(hidden_size, num_heads)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, hidden_size)
    def forward(self,q,k,v):
        out = self.crossAttention(q, k, v)
        out = self.LayerNorm(out)
        out = self.ffn(out)
        return out
    

class QFormer(nn.Module):
    def __init__(self, q_input_size=2048, kv_input_size=4096, hidden_size=2048, num_heads=64, pooled_size=1280, num_layers=2):
        super(QFormer, self).__init__()
        
        self.Qlinear = nn.Linear(q_input_size, hidden_size)
        self.Klinear = nn.Linear(kv_input_size, hidden_size)
        self.Vlinear = nn.Linear(kv_input_size, hidden_size)
        
        self.Qnorm = nn.LayerNorm(hidden_size)
        self.Knorm = nn.LayerNorm(hidden_size)
        self.Vnorm = nn.LayerNorm(hidden_size)
        self.poolnorm = nn.LayerNorm(pooled_size)
        self.linear = nn.Linear(hidden_size, q_input_size)
        self.linear_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, q_input_size)
        # self.pool_ffn = FeedForwardNetwork(Q_input_size,hidden_size)
        self.pooled = nn.Linear(hidden_size, pooled_size)
         
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(hidden_size, num_heads) for _ in range(num_layers)
        ])
        

    def forward(self, q, k, v, pool=True):
        Q_embedded = self.Qlinear(q)
        K_embedded = self.Klinear(k)
        V_embedded = self.Vlinear(v)

        Q_embedded = self.Qnorm(Q_embedded)
        K_embedded = self.Knorm(K_embedded)
        V_embedded = self.Vnorm(V_embedded)

        cross_attn_output = Q_embedded
        for cross_attn in self.cross_attn_layers:
            cross_attn_output = cross_attn(cross_attn_output, K_embedded, V_embedded)
        
        output = self.ffn(cross_attn_output)  # Apply FFN to cross-attention output
        
        if pool:
            output_pool = torch.mean(output,dim=1)
            # output_pool = self.pool_ffn(output_pool)
            output_pool = self.pooled(output_pool)

            return output, output_pool
        return output


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


class LLaMA_diffuser(nn.Module):
    def __init__(self, llama_ckpt_dir, llama_tokenizer, diffuser_ckpt_dir,
                 max_seq_len=512, max_batch_size=1,
                 v_embed_dim=2048, v_depth=4,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=77, query_layer=31,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="pretrain"):
        super().__init__()

        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc

        # 1. diffuser
        self.vae = AutoencoderKL.from_pretrained(
            diffuser_ckpt_dir, subfolder="vae"
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            diffuser_ckpt_dir, subfolder="unet"
        )
        
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            diffuser_ckpt_dir, subfolder="tokenizer", revision="", use_fast=False
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            diffuser_ckpt_dir, subfolder="tokenizer_2", revision="", use_fast=False
        )
        self.tokenizers = [self.tokenizer_one, self.tokenizer_two]

        # text encoder
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            diffuser_ckpt_dir, ""
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            diffuser_ckpt_dir, "", subfolder="text_encoder_2"
        )
        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            diffuser_ckpt_dir, subfolder="text_encoder", revision=""
        )
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            diffuser_ckpt_dir, subfolder="text_encoder_2", revision=""
        )
        self.text_encoders = [self.text_encoder_one, self.text_encoder_two]

        # noise_scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            diffuser_ckpt_dir, subfolder="scheduler"
        )

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. query, blocks and projector
        self.query_embedding = nn.Embedding(query_len, v_embed_dim)

        self.query_block = QFormer(
            q_input_size=2048,
            kv_input_size=4096,
            pooled_size=1280,
        )
        
        # 3. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 4. llama
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

         # 5. training criterion
        self.criterion = torch.nn.MSELoss()
        
        # 6. training parameters
        self.phase = phase
        self.get_trainable_params(self.phase)

        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self, phase='pretrain'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['query_embedding', 'query_block']
            for name, para in self.named_parameters():            
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
        
        else:
            raise ValueError(f"Unknown model phase: {phase}")

    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents


    def forward(self, tokens, labels, imgs, prompts, original_sizes, crops_coords_top_lefts): 
        # sdxl text embeddings
        prompt_embeds_list = []
        with torch.no_grad():
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        sdxl_prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        sdxl_pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)         
    

        bsz, seqlen = tokens.shape
       
        llama_text_embeddings = self.llama(tokens)

        # cross attention between query and llama text embedding
        query_embedding = self.query_embedding.weight.unsqueeze(0).repeat(len(imgs), 1, 1)
        
        prompt_embeds, pooled_prompt_embeds = self.query_block(query_embedding, llama_text_embeddings, llama_text_embeddings)
        
        # convert images to latent sapce
        # vae.cuda().encode(torch.randn((1, 3, 224, 224), dtype=torch.float32).cuda()).latent_dist.sample()
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                model_input = self.vae.encode(imgs).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()
       
        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        add_time_ids = torch.cat(
            [self.compute_time_ids(original_sizes[i], crops_coords_top_lefts[i], device=model_input.device, dtype=model_input.dtype) for i in range(bsz)]
        )

        # preduct the noise residual
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_prompt_embeds
        }
        
        model_pred = self.unet(noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = model_input
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        text1_loss = F.mse_loss(prompt_embeds.float(), sdxl_prompt_embeds.float(), reduction="mean")
        text2_loss = F.mse_loss(pooled_prompt_embeds.float(), sdxl_pooled_prompt_embeds.float(), reduction="mean")
        
        return loss, text1_loss, text2_loss
        #return 0.5 * text1_loss, 0.5 * text1_loss, text2_loss

    # time ids
    def compute_time_ids(self, origin, crops, device, dtype):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (int(origin[0]), int(origin[1]))
        crops_coords_top_left = (int(crops[0]), int(crops[1]))
        target_size = (512, 512)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(device, dtype=dtype)

        return add_time_ids

    @torch.inference_mode()
    def generate(
        self, prompts, num_inference_steps=25, seed=1234, 
        do_classifier_free_guidance=True, guidance_scale=5.0,
        guidance_rescale=0.0, use_origin=False, use_sdxl_text2=False, use_sdxl_text1=False,
    ):
        bsz = len(prompts)
        num_images_per_prompt = 1
        params = self.llama.params
        width = 1024
        height = 1024

        
        prompt_embeds_list = []
        with torch.no_grad():
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        sdxl_prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        sdxl_pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        # use sdxl original
        prompt_embeds = sdxl_prompt_embeds.cuda()
        pooled_prompt_embeds = sdxl_pooled_prompt_embeds.cuda()
        
        # 1. Encode input prompt
        if isinstance(prompts[0], str):
            tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in prompts]
        tokens = torch.tensor(tokens).cuda().long()
        llama_text_embeddings = self.llama(tokens)

        # cross attention between query and llama text embedding
        query_embedding = self.query_embedding.weight.unsqueeze(0).repeat(bsz, 1, 1)
        
        if not use_origin:
            prompt_embeds, pooled_prompt_embeds = self.query_block(query_embedding, llama_text_embeddings, llama_text_embeddings)

        negative_prompt_embeds = torch.zeros_like(prompt_embeds, dtype=prompt_embeds.dtype)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds, dtype=prompt_embeds.dtype)

        # 2. Prepare timesteps
        device = prompt_embeds.device
        #dtype = prompt_embeds.dtype
        dtype = torch.float32

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        # 3. Prepare latent variables
        generator = torch.Generator(device=prompt_embeds.device).manual_seed(seed)
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            bsz * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )

        # 4. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds

        add_time_ids = torch.cat(
            [self.compute_time_ids((1024, 1024), (0, 0), device=device, dtype=dtype) for i in range(bsz)]
        )

        negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)
        extra_step_kwargs = {"generator": None}
        cross_attention_kwargs = None

        # 5. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.noise_scheduler.order, 0)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            # make sure the VAE is in float32 mode, as it overflows in float16
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        with torch.cuda.amp.autocast(enabled=False):
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        image = self.image_processor.postprocess(image, output_type='pil')
        return image
    
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    "LORA-BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth",
    "CAPTION-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())


def load(name, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}"), None

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
    diffuser_ckpt_dir = ""

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA_diffuser from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_diffuser(
        llama_ckpt_dir, llama_tokenzier_path, diffuser_ckpt_dir,
        max_seq_len=512, max_batch_size=1,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        phase=phase)

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device)



