import os
import torch
import cv2
import random
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=)


imgs1 = [torch.from_numpy(cv2.resize(cv2.imread(os.path.join("../coco_result", img)), (512, 512))) for img in random.sample(os.listdir("../coco_result/"), 3000)]
imgs1 = [img.permute(2, 0, 1) for img in imgs1]
imgs1 = torch.stack(imgs1, dim=0)
fid.update(imgs1, real=False)


imgs2 = [torch.from_numpy(cv2.resize(cv2.imread(os.path.join("/data0/xiaowei/val2014", img)), (512, 512))) for img in random.sample(os.listdir("/data0/xiaowei/val2014"), 3000)]
imgs2 = [img.permute(2, 0, 1) for img in imgs2]
imgs2 = torch.stack(imgs2, dim=0)
fid.update(imgs2, real=True)

fid.compute()
