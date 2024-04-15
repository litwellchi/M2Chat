import os
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download

from .sd_qformer import SDQFormer

import diffusers
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection
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
# from peft import LoraConfig, set_peft_model_state_dict


class LLaMA_mmdiffuser(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer, diffuser_ckpt_dir,
                 max_seq_len=512, max_batch_size=1,
                 clip_model='ViT-L/14',
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="finetune"):
        super().__init__()

        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc

        # 0.1.1 diffuser
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

        # 0.1.2 text encoder
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            diffuser_ckpt_dir, subfolder="text_encoder", revision=""
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            diffuser_ckpt_dir, subfolder="text_encoder_2", revision=""
        )
        self.text_encoders = [self.text_encoder_one, self.text_encoder_two]

        #0.2 noise_scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            diffuser_ckpt_dir, subfolder="scheduler"
        )

        # 0.3 sd qformer
        self.sd_query = nn.Embedding(77, 2048)
        self.query_block = QFormer(
            q_input_size=2048,
            kv_input_size=4096,
            pooled_size=1280,
        )
        

        # 1. clip and clip projector
        self.clip, self.clip_transform = clip.load(clip_model)

        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. visual query, blocks and projector
        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama
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

        del self.clip.transformer

         # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # 7. training parameters
        print('Check the trainable params of LLAMA-mmdiffuser')
        self.phase = phase
        self.get_trainable_params(self.phase)

        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
               
        # 8. Adding LoRA
        # unet_lora_config = LoraConfig(
        #     r=lora_rank,
        #     lora_alpha=lora_rank,
        #     init_lora_weights="gaussian",
        #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        # )
        # unet.add_adapter(unet_lora_config)

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            # train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'adapter_query','sd_query', 'query_block']
            # train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query','sd_query', 'query_block']
            # train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query','sd_query', 'query_block']
            train_param_name = ['sd_query', 'query_block']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
                        # print(name)
                    else:
                        para.requires_grad = False
        
        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
        
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        
    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, imgs):
        clip_feats = self.clip_encode_image(imgs)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = self.visual_query.weight.unsqueeze(
            0).repeat(len(imgs), 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    def forward(self, tokens, labels, imgs, prompts, original_sizes, crops_coords_top_lefts, ref_image=None, img_labels=None): 
        # sdxl text embeddings
        prompt_embeds_list = []
        try:
            sd_prompts = [prompt.split("### Response", 1)[1] for prompt in prompts]
            sd_prompts = [sd_prompt.split("</IC>")[1] for sd_prompt in sd_prompts]
            sd_prompts = [sd_prompt.replace("<|img|>","") for sd_prompt in sd_prompts]
        except:
            sd_prompts = prompts
        
        with torch.no_grad():
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                text_inputs = tokenizer(
                    sd_prompts,
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
        # llama_text_embeddings = self.llama(tokens, return_token)

        # TODO only for bsz=1
        # if img_labels is not None:
        #     pos_count = torch.nonzero(img_labels == 0).size(0)
        # else:
        #     pos_count = torch.nonzero(labels == 0).size(0)
        
        querry_llama_text_embeddings = torch.zeros_like(llama_text_embeddings[:,0:32,:])
        # print("querry_llama_text_embeddings shape:",querry_llama_text_embeddings.shape)
        for i in range(llama_text_embeddings.shape[0]):            
            if img_labels is not None:
                pos_count = torch.nonzero(img_labels[i,:] == 0).size(0)
            else:
                pos_count = torch.nonzero(labels[i,:] == 0).size(0)
            # print("img_labels.shape",img_labels.shape)
            # print("feature",img_labels[i,:])

            try:
                querry_llama_text_embeddings[i,:,:] = querry_llama_text_embeddings[i,:,:]+llama_text_embeddings[i,pos_count:pos_count+32,:]
            except:
                print("pos count,", pos_count)
                print("llama_text_embeddings[i,:,:] shape:",llama_text_embeddings[i,:,:].shape)
                print("querry_llama_text_embeddings shape[i,:,:]:",querry_llama_text_embeddings[i,:,:].shape)
                print("querry_llama_text_embeddings[i,:,:]",querry_llama_text_embeddings[i,:,:])
                print("llama_text_embeddings[i,pos_count:pos_count+32,:] shape:",llama_text_embeddings[i,pos_count:pos_count+32,:].shape)
                print("llama_text_embeddings[i,pos_count:pos_count+32,:]",llama_text_embeddings[i,pos_count:pos_count+32,:])
                querry_llama_text_embeddings[i,:,:] = querry_llama_text_embeddings[i,:,:]+llama_text_embeddings[i,-32:,:]
                # exit(0)
        # cross attention between query and llama text embedding
        query_embedding = self.sd_query.weight.unsqueeze(0).repeat(len(imgs), 1, 1)
        prompt_embeds, pooled_prompt_embeds = self.query_block(query_embedding, querry_llama_text_embeddings, querry_llama_text_embeddings)
        
        # convert images to latent sapce
        # vae.cuda().encode(torch.randn((1, 3, 224, 224), dtype=torch.float32).cuda()).latent_dist.sample()
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                if ref_image == None: ref_image = imgs
                model_input = self.vae.encode(ref_image).latent_dist.sample()
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
        
        
        h = self.llama.norm(llama_text_embeddings)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            t_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            t_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
        # print(t_loss)
        return loss, t_loss, text1_loss, text2_loss
        #return 0.5 * text1_loss, 0.5 * text1_loss, text2_loss

    @torch.inference_mode()
    def t2i_generate(
        self, prompts, num_inference_steps=25, seed=1234, 
        do_classifier_free_guidance=True, guidance_scale=5.0,
        guidance_rescale=0.0, use_origin=False, use_sdxl_text2=False, use_sdxl_text1=False,
    ):
        bsz = len(prompts)
        num_images_per_prompt = 1
        params = self.llama.params
        width = 1024
        height = 1024

        try:
            sd_prompts = [prompt.split("###Response", 1)[1] for prompt in prompts]
            print(sd_prompts)
            sd_prompts = [prompt.replace("<|img|>") for prompt in prompts]
        except:
            sd_prompts = prompts
        
        prompt_embeds_list = []
        with torch.no_grad():
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                text_inputs = tokenizer(
                    sd_prompts,
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
        
        if not use_origin:
            # 1. Encode input prompt
            if isinstance(prompts[0], str):
                tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in prompts]
            tokens = torch.tensor(tokens).cuda().long()
            llama_text_embeddings = self.llama(tokens)
            
            # cross attention between query and llama text embedding
            query_embedding = self.sd_query.weight.unsqueeze(0).repeat(bsz, 1, 1)
            llama_text_embeddings = llama_text_embeddings.to(query_embedding.dtype)
            
            # print(prompts)
            # labels = [prompt.split("###Response", 1)[1] for prompt in prompts]
            prompts = [x.split("### Response", 1)[0] for x in prompts]
            label_tokens = [([self.tokenizer.encode(x, bos=True, eos=True) for x in prompts])]
            # print(label_tokens[0] )
            pos_count = len(label_tokens[0])
            querry_llama_text_embeddings = llama_text_embeddings[:,pos_count:pos_count+128,:]
            prompt_embeds, pooled_prompt_embeds = self.query_block(query_embedding, querry_llama_text_embeddings, querry_llama_text_embeddings)

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
    

    def ti2t_forward(self, tokens, labels, imgs):
        visual_query = self.forward_visual(imgs)

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        return c_loss, c_loss

    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int, return_hidden=False):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        hidden = h
        
        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        if return_hidden:
            return output.float(), hidden.float()
        return output.float()

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
    def ti2ti_generate(
        self, imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,num_inference_steps=25, seed=1234, 
        do_classifier_free_guidance=True, guidance_scale=5.0,
        guidance_rescale=0.0, use_origin=False, use_sdxl_text2=False, use_sdxl_text1=False,
    ):
        # TODO
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)
        raw_prompts = prompts

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()


        # text generation
        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # image generation 
        # with torch.cuda.amp.autocast():
        #     _, llama_text_embeddings = self.forward_inference(visual_query, tokens[:, prev_pos:min_prompt_size], prev_pos, return_hidden=True)

        # Text Generation
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        
        # Image Generation
        raw_prompts = [raw_prompts[0]+decoded[0]]
        num_images_per_prompt = 1
        params = self.llama.params
        width = 1024
        height = 1024
        
        
        if isinstance(raw_prompts[0], str):
            tmp_prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in raw_prompts]
        min_prompt_size = min([len(t) for t in tmp_prompts])
        max_prompt_size = max([len(t) for t in tmp_prompts])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)        
        all_gen_token = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        # text generation
        for k, t in enumerate(tmp_prompts):
            all_gen_token[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = all_gen_token != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # image generation 
        with torch.cuda.amp.autocast():
            _, llama_text_embeddings = self.forward_inference(visual_query, all_gen_token[:, prev_pos:min_prompt_size], prev_pos, return_hidden=True)
        # llama_text_embeddings = self.llama(all_gen_token)    
        query_embedding = self.sd_query.weight.unsqueeze(0).repeat(bsz, 1, 1)
        llama_text_embeddings = llama_text_embeddings.to(query_embedding.dtype)
        
        response = []
        for raw_prompt in raw_prompts:        
            if "</IC>" in raw_prompt:
                response.append(raw_prompt.split("</IC>", 1)[0])
            else:
                response.append(raw_prompt.split("### Response", 1)[0])
        # try:
        #     sd_prompts = [prompt.split("### Response", 1)[1] for prompt in prompts]
        #     sd_prompts = [sd_prompt.split("</IC>")[1] for sd_prompt in sd_prompts]
        #     sd_prompts = [sd_prompt.replace("<|img|>","") for sd_prompt in sd_prompts]
        # except:
        #     sd_prompts = prompts
        label_tokens = [([self.tokenizer.encode(x, bos=True, eos=True) for x in response])]
        # print(label_tokens[0] )
        pos_count = len(label_tokens[0][0])
        querry_llama_text_embeddings = llama_text_embeddings[:,pos_count:pos_count+32,:]
        
        prompt_embeds, pooled_prompt_embeds = self.query_block(query_embedding, querry_llama_text_embeddings, querry_llama_text_embeddings)

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
        return decoded, image

    @torch.inference_mode()
    def generate(
        self, imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded

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

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_mmdiffuser(
        llama_ckpt_dir, llama_tokenzier_path,
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
    return model.to(device), model.clip_transform