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

GENERATION_INSTRUCTION = 'generate next image'

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

class VIST_Finetune_Dataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")        
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        data = []
        images, captions = [], []
        for meta_path in self.config['META']:
            with open(meta_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
            # Convert the reader object into a list of lists
                data.extend(list(reader))
                print(f"{meta_path}: len {reader.line_num}")
        
        self.ann = data
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)
    


    def __getitem__(self, index):
        # sample = self.data_list[index]
        data_item = self.ann[index] #[image_path, caption]
        caption = data_item[1]
        
        # read image
        try:
            filename = data_item[0]
            caption = data_item[1]
            tar_path, image_path = os.path.split(filename)
            tar = tarfile.open(tar_path, 'r')
            image_file = tar.extractfile(image_path)
            image_data = image_file.read()
            image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
     
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        except:
            return self.__getitem__(index+1)
        
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
        # input1 = caption
        format_instruction = GENERATION_INSTRUCTION
        format_input = caption
        answer = caption
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image, caption, torch.tensor(original_size), torch.tensor(crop_top_left)
    
# Preprocessing the datasets.
train_resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)
train_crop = transforms.CenterCrop(256)
train_flip = transforms.RandomHorizontalFlip(p=1.0)
train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
