'''
THIS DATA PROCESS FILE IS FOR STAT-641 W2021 FINAL PROJECT

Author: Daniel Yang
'''


#excuse the lack of PEP-8

from bs4 import BeautifulSoup
import cv2
import numpy as np 
import pandas as pd 
import xml.etree.ElementTree as ET
import os
import os.path
from glob import glob
#our data is contained in face-mask-detection 
#folder has "annotations (in xml)", "images", and "labels (empty)"
#this script writes xml to txt and divides into testing/training/validation

data_path = 'C:/Users/Daniel/stat-641/face-mask-detection/labels'
#os.mkdir(data_path)


def xml_to_txt(path):

    root = ET.parse(path)
    img_path = root.find('filename').text.replace('png', 'txt')
    label = os.path.join(data_path, img_path)
    with open(label, "w") as f:
        text = []
        for node in root.getroot().iter('size'):
            width = int(node.find('width').text)
            height = int(node.find('height').text)
        
        for node in root.getroot().iter('object'):
            mask_type = node.find('name').text
            if (mask_type == "without_mask"):
                classification = '0'
            elif (mask_type == "with_mask"):
                classification = '1'
            else:
                classification = '2'
            xmin = int(node.find('bndbox/xmin').text)
            xmax = int(node.find('bndbox/xmax').text)
            ymin = int(node.find('bndbox/ymin').text)
            ymax = int(node.find('bndbox/ymax').text)

            dw = 1/width
            dh = 1/height
            x_center = str(((xmin+xmax)/2-1) * dw)
            y_center = str(((ymin+ymax)/2-1)* dh)
            w_norm = str((xmax - xmin) * dw)
            h_norm = str((ymax - ymin) * dh)

            line = classification + ' ' + x_center + ' ' + y_center + ' ' + w_norm + ' ' + h_norm
            text.append(line)
            text.append('\n')

        for line in text[:-1]:
            f.write(line)
    f.close






#writes txt files corresponding to training and validation
#
def test_train_split():
    splits = []
    img_path ='C:/Users/Daniel/stat-641/face-mask-detection/images'
    img_path =  os.listdir(img_path)
    for img in img_path:
        splits.append(os.path.join('/content/STAT-641-PROJECT/face-mask-detection/images', img)) # google colab

    np.random.shuffle(splits)

    #partition dataset into .7, .2, .1 
    train, valid, test = np.split(splits, [int(len(splits)*0.7), int(len(splits)*0.9)])
    data_path = 'C:/Users/Daniel/stat-641/face-mask-detection/'
    with open(os.path.join(data_path, "train.txt"), 'w') as f:
        lines = list('\n'.join(train))
        f.writelines(lines)
    f.close

    with open(os.path.join(data_path, "valid.txt"), 'w') as f:
        lines = list('\n'.join(valid))
        f.writelines(lines)
    f.close

    # write test.txt
    with open(os.path.join(data_path, "test.txt"), 'w') as f:
        lines = list('\n'.join(test))
        f.writelines(lines)
    f.close


def data_process():
    test_train_split() 
    #label_dir = list(sorted(glob("C:/Users/Daniel/stat-641/face-mask-detection/annotations/*.xml")))
    #for file in label_dir:
    #    xml_to_txt(file)

# a bunch of random checks go down here        

#labels_dir = list(sorted(glob("C:/Users/Daniel/stat-641/face-mask-detection/annotations/*.xml")))
#print(labels_dir)
#print(os.path.exists('C:/Users/Daniel/stat-641/face-mask-detection/images'))

print(os.path.exists(data_path))
print(data_path)
data_process()
