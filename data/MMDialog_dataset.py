import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import cv2
import os
import csv
import tarfile
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

GENERATION_INSTRUCTION = 'answer based on the input dialog'

def format_mmDialog_sprompt(instruction='answer based on the input dialog', turn1_context=None, turn1_caption=None):
    # input = "A said" + turn1_context + "and post an image about "+ turn1_caption
    input = "A said" + turn1_context + "and post an image about " + turn1_caption
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
    }
    return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

class mmdialog_Finetune_Dataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")        
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        data = []
        images, captions = [], []
        for meta_path in self.config['META']:
            with open(meta_path,'r') as file:
                folder_path = meta_path.replace("conversations_pair_catpion_0.json",'')
                reader = json.loads(file.read())
                reader = [[folder_path,i] for i in reader if len(i)>1]
            # with open(meta_path, 'r') as csv_file:
            #     reader = csv.reader(csv_file)
            # Convert the reader object into a list of lists
            
                data.extend(list(reader))
                # print(f"{meta_path}: len {data}")
        
        self.ann = data
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)
    


    def __getitem__(self, index):
        # sample = self.data_list[index]
        data_item = self.ann[index][1] #[image_path, caption]
        folder_path = self.ann[index][0] #[image_path, caption]
        # caption = data_item[1]
        
        # read image
        # simplified version
        try:
            round1 = data_item[0]
            round2 = data_item[1]
            context1 = round1['turn'][0]["__TEXT__"]
            context2 = round2['turn'][0]["__TEXT__"]
            inputimg_dir = os.path.join(folder_path,round1['turn'][1]["__MEDIA__"]+'.jpg')
            gen_image = cv2.imread(os.path.join(folder_path,round2['turn'][1]["__MEDIA__"]+'.jpg'))
            
            for i in range(len(round1['turn'])):
                if "caption" in round1['turn'][i].keys():
                    caption1 = round2['turn'][i]["caption"]
            
            for i in range(len(round2['turn'])):
                if "caption" in round2['turn'][i].keys():
                    caption2 = round2['turn'][i]["caption"]

            image = cv2.imread(inputimg_dir)
            image = Image.fromarray(image)
            gen_image = Image.fromarray(gen_image)
        except:
            # exit(0)
            if index+1 < len(self.ann):
                return self.__getitem__(index+1)
            else:
                index = 0
                return self.__getitem__(index)
        
        original_size = (image.height, image.width)
        # resize 
        image = train_resize(image)

        # crop
        y1 = max(0, int(round((image.height - 256) / 2.0)))
        x1 = max(0, int(round((image.width - 256) / 2.0)))
        image = train_crop(image)

        if random.random() < 0.5:
            # flip
            x1 = image.width - x1
            image = train_flip(image)
        
        crop_top_left = (y1, x1)
        # transform
        image = train_transforms(image)
        
        
        # resize 
        gen_image = train_resize(gen_image)

        # crop
        y1 = max(0, int(round((gen_image.height - 256) / 2.0)))
        x1 = max(0, int(round((gen_image.width - 256) / 2.0)))
        gen_image = train_crop(gen_image)

        if random.random() < 0.5:
            # flip
            x1 = gen_image.width - x1
            gen_image = train_flip(gen_image)
        
        crop_top_left = (y1, x1)
        # transform
        gen_image = train_transforms(gen_image)
        
        # input1 = caption
        format_instruction = GENERATION_INSTRUCTION
        caption1 = ""
        input1 = format_mmDialog_sprompt(format_instruction, context1, caption1)
        answer = context2+" </IC>" + caption2 + "</IC>" + "<|img|>"*128
        input2 = input1 + answer
        image_prompt_input = input1 + context2+" </IC>"
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        image_prompt_input = torch.tensor(self.tokenizer.encode(image_prompt_input, bos=True, eos=False), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]

        labels = copy.deepcopy(input2)
        img_labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        img_labels[:len(image_prompt_input)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        img_label_mask = img_labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        img_labels[~img_label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        # print("img label",img_labels)
        # print("label",labels.shape)
        # print(labels)
        # exit(0)
        return input2, labels, input2_mask, image, context2, torch.tensor(original_size), torch.tensor(crop_top_left), gen_image, img_labels
    
# Preprocessing the datasets.
train_resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)
train_crop = transforms.CenterCrop(256)
train_flip = transforms.RandomHorizontalFlip(p=1.0)
train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
