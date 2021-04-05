#DOESN'T WORK DON'T KNOW WHY

from glob import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from bs4 import BeautifulSoup
import cv2
import os
from CustomDataSet import FaceMaskData 
from torchvision.utils import save_image

imgs_dir = list(sorted(glob('C:/Users/Daniel/stat-641/face-mask-detection/images/*.png'))) #replace with any directory of your choosing
dataset = FaceMaskData(csv_file = "csvlabels.csv", imgs = imgs_dir) #note that you need the csvlabels 

#print(os.path.exists("C:/Users/Daniel/stat-641/face-mask-detection/images"))
'''
#You run this, your computer will be destroyed
img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'img' + str(img_num)+'.png')
        img_num += 1 

'''