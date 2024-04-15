import shutil
import os

imgs = os.listdir("coco_cc3m_result")

for img in imgs:
    img_name = int(img[:-4]) + 5000
    print(img_name)
    shutil.copy(os.path.join("coco_cc3m_result", img), os.path.join("coco_cc3m_result", str(img_name) + ".jpg"))
