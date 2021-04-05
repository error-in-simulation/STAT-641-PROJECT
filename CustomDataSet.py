#custom dataset for RCNN
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import cv2

class FaceMaskData(Dataset):
    def __init__(self, csv_file, imgs):
        self.annotations = pd.read_csv(csv_file)
        self.imgs = imgs
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness = 0.5),
            transforms.RandomRotation(degrees = 45),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()

        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image = cv2.imread(self.imgs[index])
        image = self.transform(image)
        labels = torch.tensor(int(self.annotations.iloc[index, 1]))
        bboxes = self.annotations.iloc[index, 2:]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        image_id = torch.tensor([index])
        


        target = {}
        target["boxes"] = bboxes 
        target["labels"] = labels
        target["image_id"] = image_id


        return image, target 


