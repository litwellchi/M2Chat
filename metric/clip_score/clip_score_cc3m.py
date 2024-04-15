import os
import cv2
import torch
import numpy as np
import csv

from torchmetrics.multimodal.clip_score import CLIPScore

metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

metric = metric.cuda()

score_list = []

with open('/data1/linziyi/cc3m.csv', newline='') as csvfile:

   coco_txts = csv.reader(csvfile, delimiter=' ', quotechar='|')

   for i, txt in enumerate(coco_txts):
        if i == 0 or i == 1370 or i == 1378:
            continue
        txt = " ".join(txt).split(",conceptual_captions")[0]
        if not os.path.exists(os.path.join("../../coco_cc3m_result", str(i)+".jpg")):
            break
        img = cv2.imread(os.path.join("../../coco_cc3m_result", str(i)+".jpg"))
        print(i, txt)
        score = metric(torch.from_numpy(img).cuda(), txt)
        if score.detach().cpu():
            score = score.detach().cpu().numpy()    
            score_list.append(score)

print(len(score_list), np.array(score_list).mean())
