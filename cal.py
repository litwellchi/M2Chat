import os
import cv2
img_list = os.listdir("coco_result/")
img_save = []
for img in img_list:
    img_path = os.path.join("coco_result/", img)
    img = cv2.imread(img_path)
    print(img_path, img.shape, img.shape[2])
    if img.shape[2] != 3:
        img_save.append(img_path)

print(img_save)
