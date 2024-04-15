import os
import cv2
import torch
import numpy as np

from torchmetrics.multimodal.clip_score import CLIPScore

metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

metric = metric.cuda()
coco_text = open("../../coco_30k.txt").readlines()
res_path = "../../coco_result"

score_list = []

for i, txt in enumerate(coco_text):
    if os.path.exists(os.path.join(res_path, str(i)+".jpg")):
        img = cv2.imread(os.path.join(res_path, str(i)+".jpg"))
        score = metric(torch.from_numpy(img).cuda(), txt)
        score = score.detach().cpu().numpy()    
        print(score)
        score_list.append(score)

print(len(score_list), np.array(score_list).mean())
