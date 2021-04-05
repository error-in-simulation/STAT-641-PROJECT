#custom dataset for RCNN
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io


class FaceMaskData(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        labels = torch.tensor(int(self.annotations.iloc[index, 1]))
        bboxes = self.annotations.iloc[index, 2:]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        image_id = torch.tensor([index])
        


        target = {}
        target["image"] = image
        target["boxes"] = bboxes 
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target 


