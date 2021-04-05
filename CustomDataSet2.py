#customdataset v2

from glob import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from bs4 import BeautifulSoup
import cv2
import os


imgs_dir = list(sorted(glob('C:/Users/Daniel/stat-641/face-mask-detection/images/*.png')))
labels_dir = list(sorted(glob("C:/Users/Daniel/stat-641/face-mask-detection/annotations/*.xml")))

class dataset(Dataset) :
    def __init__(self, imgs, labels) :
        self.imgs = imgs
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        
    def __len__(self) :
        return len(self.imgs)
    
    def __getitem__(self, index) :
        x = cv2.imread(self.imgs[index])
        x = self.transform(x).to(self.device)
        
        y = dict()
        with open(self.labels[index]) as f :
            data = f.read()
            soup = BeautifulSoup(data, 'xml')
            data = soup.find_all('object')
            
            box = []
            label = []
            for obj in data :
                xmin = int(obj.find('xmin').text)
                ymin = int(obj.find('ymin').text)
                xmax = int(obj.find('xmax').text)
                ymax = int(obj.find('ymax').text)
                
                label_ = 0
                if obj.find('name').text == 'with_mask' :
                    label_ = 1
                elif obj.find('name').text == 'mask_weared_incorrect' :
                    label_ = 2
                
                box.append([xmin, ymin, xmax, ymax])
                label.append(label_)
                
            box = torch.FloatTensor(box)
            label = torch.IntTensor(label)
            
            y['image_id'] = torch.FloatTensor([index]).to(device)
            y["boxes"] = box.to(device)
            y["labels"] = torch.as_tensor(label, dtype=torch.int64)
            
        return x, y
    


data = dataset(imgs = imgs_dir, labels = labels_dir)

print(data.__getitem__(1))

#print(imgs_dir)