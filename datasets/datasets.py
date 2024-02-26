from .utils import *
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
import os
from math import *

from torch.utils.data import Dataset, DataLoader

class MPII(Dataset):
    def __init__(self, split, inputRes, outputRes, maxScale = 0.25, 
                 maxRotate = 30, hmGauss = 1):
        
        annotation_file = None
        self.split = split
        self.maxScale = maxScale
        self.maxRotate = maxRotate
        self.hmGauss = hmGauss
        
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.nJoints = 16
        
        if self.split == 'train':
            annotation_file = './datasets/train.csv'
        elif self.split == 'valid':
            annotation_file = './datasets/valid.csv'
        elif self.split == 'test':
            annotation_file = './datasets/test.csv'

        self.annotation = pd.read_csv(annotation_file)

    def __len__(self):
        return self.annotation.shape[0]

    def getElement(self, instance):
        imgPath = os.path.join(os.getcwd(), 'datasets' ,instance['image_path'])
        pts = instance['part']
        c = instance['center']
        s = instance['scale']

        pts = np.fromstring(pts.replace('[', '').replace(']', '').replace('\n', ''), sep=' ').reshape(-1, 2)
        c = np.fromstring(c.replace('[', '').replace(']', '').replace('\n', ''), sep=' ').reshape(2)
        
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        s = s * 200
        
        return imgPath, pts, c, s

    def plot_image(self, idx, color = (255, 0, 0)):
        instance = self.annotation.iloc[idx]
        img_path, pts, c, s = self.getElement(instance)

        # print(s, c)
        r = 0
        
        img = cv2.imread(img_path)

        inp = Crop(img, c, s, r, self.inputRes) / 256
        inp = np.transpose(inp, (1, 2, 0))
        
            
        # out = np.zeros((self.nJoints, self.outputRes, self.outputRes))
        
        if (self.split == 'train' or self.split == 'valid'):
            for i in range(self.nJoints):
                pts[i] = Transform(pts[i], c, s, r, self.outputRes)             
                pts[i] = pts[i] * self.inputRes / self.outputRes + (self.inputRes / self.outputRes) / 2 

                x, y = int(pts[i][0]), int(pts[i][1])
                cv2.circle(inp, (x, y), 2, color= color, thickness = 2)
                
            plt.imshow(inp)
            plt.axis('off')
            plt.show()
    
        else: 
            print("Only Training and Valid set")
    
    def __getitem__(self, idx):
        
        instance = self.annotation.iloc[idx]
        img_path, pts, c, s = self.getElement(instance)

        r = 0
        
        img = cv2.imread(img_path)
        inp = Crop(img, c, s, r, self.inputRes) / 256
        
        out = np.zeros((self.nJoints, self.outputRes, self.outputRes))
        
        if (self.split == 'train' or self.split == 'valid'):
            for i in range(self.nJoints):
                pts[i] = Transform(pts[i], c, s, r, self.outputRes)             
                out[i] = DrawGaussian(out[i], pts[i], self.hmGauss, 
                                      0.5 if self.outputRes == 32 else -1)

        
        return inp, pts, out