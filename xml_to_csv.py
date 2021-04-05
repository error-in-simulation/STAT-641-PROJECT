#xml to CSV 
#Seems easier to work with data loaders using CSV, so this is already ran

from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xml.etree.ElementTree as ET
import csv
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset

#idea is to convert xml to pandas to csv I guess

annotations = list(sorted(glob("C:/Users/Daniel/stat-641/face-mask-detection/annotations/*.xml")))

df = []


for file in annotations:
    row = []
    tree = ET.parse(file)
    filename = tree.find('filename').text.replace('xml', 'png')
    for node in tree.getroot().iter('object'):
        mask_type = node.find('name').text
        if mask_type == 'without_mask':
            mask = 0
        elif mask_type == 'with_mask':
            mask = 1
        else:
            mask = 2
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = int(node.find('bndbox/ymin').text)
        ymax = int(node.find('bndbox/ymax').text)

        row = [filename, mask, xmin, ymin, xmax, ymax]
        df.append(row)

data = pd.DataFrame(df, columns=['filename', 'mask' ,'xmin', 'ymin', 'xmax', 'ymax'])
print(data.head(10))

data = data.to_csv('csvlabels.csv', index=False)


example_frame = pd.read_csv('csvlabels.csv')

