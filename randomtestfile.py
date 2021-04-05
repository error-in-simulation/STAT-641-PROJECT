#DOESN'T WORK DON'T KNOW WHY

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from CustomDataSet import FaceMaskData
import matplotlib.pyplot as plt
import os
import glob 
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness = 0.5),
    transforms.RandomRotation(degrees = 45),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomGrayscale(p=0.2)

])
#image_dir = list(sorted(glob('C:/Users/Daniel/stat-641/face-mask-detection/images/*.png')))
dataset = FaceMaskData(csv_file = "csvlabels.csv", root_dir ='C:/Users/Daniel/stat-641/face-mask-detection/images', transform = my_transforms)

#print(os.path.exists("C:/Users/Daniel/stat-641/face-mask-detection/images"))


print(dataset.__getitem__(1))