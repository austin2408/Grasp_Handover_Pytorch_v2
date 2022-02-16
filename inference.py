import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torch.utils import model_zoo
import random
import sys
import math

import torchvision.transforms as transforms

import cv2
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
from model import *
from Dataloader import *
import argparse

net = GraspNet(4)
net = net.cuda()
net.load_state_dict(torch.load('/home/austin/HANet_v2/grapnet_50_0.001184779648870712.pth'))
net.eval()

dataset = parallel_jaw_based_grasping_dataset('/home/austin/Datasets/FCN_Aff', mode='test')
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 8)

col = 9
show = True
y = 0
Data = [0,0,0,0]
pred = [0,0,0,0]
A = ['90','-45','0','45']

for i, batch in enumerate(dataloader):
    if i >= 5:
        break
    else:
        color = batch['color'].cuda()
        depth = batch['depth'].cuda()
        label = batch['label'].permute(0,2,3,1)
        angle = batch['id'][0].item()

        start = time.time()

        predict = net(color, depth)
        predict = predict.cpu().detach().numpy()

        D = batch['depth'][0].cpu().detach().numpy()[0]
        C = batch['origin_color'].cpu().detach().numpy()[0]
        Max = []
        Re = []
        re = np.zeros((4, 224, 224))

        for i in range(4):
            x, y = np.where(predict[0][i] == np.max(predict[0][i]))
            re[i] = cv2.resize(predict[0][i], (224, 224))
            # re[i] = cv2.circle(re[0], (y[0]*8, x[0]*8), 10, 4)
            Max.append(np.max(predict[0][i]))
            Re.append(re[i])

        graspable = re[Max.index(max(Max))]
        graspable [D==0] = 0
        graspable[graspable>=1] = 0.99999
        graspable[graspable<0] = 0
        graspable = cv2.GaussianBlur(graspable, (7, 7), 0)
        affordanceMap = (graspable/np.max(graspable)*255).astype(np.uint8)
        affordanceMap = cv2.applyColorMap(affordanceMap, cv2.COLORMAP_JET)
        affordanceMap = affordanceMap[:,:,[2,1,0]]

        C = cv2.addWeighted(C,0.7,affordanceMap, 0.3,0)
        x, y = np.where(predict[0][Max.index(max(Max))] == np.max(predict[0][Max.index(max(Max))]))

        print("Computing time : ",time.time() - start)
        # C = cv2.circle(C, (y[0]*8, x[0]*8), 2,(0,255,0), 4)




        Data[angle] += 1
        if (Max.index(max(Max)) == angle):
            pred[Max.index(max(Max))] += 1
            y += 1

        if show == True:
            plt.title('Angle : '+A[Max.index(max(Max))]+', Position : ('+str(y[0]*8)+','+str(x[0]*8)+')')
            plt.imshow(C)

            # fig = plt.figure(figsize=(10, 10))
            # fig.add_subplot(1, col, 1)
            # plt.title('90')
            # plt.imshow(Re[0])

            # fig.add_subplot(1, col, 2)
            # plt.title('-45')
            # plt.imshow(Re[1])
            # # plt.imshow(out[0][0])

            # fig.add_subplot(1, col, 3)
            # plt.title('0')
            # plt.imshow(Re[2])

            # fig.add_subplot(1, col, 4)
            # plt.title('45')
            # plt.imshow(Re[3])

            # fig.add_subplot(1, col, 5)
            # plt.title('color')
            # plt.imshow(C)

            # fig.add_subplot(1, col, 6)
            # plt.title('color')
            # plt.imshow(label[0][0].cpu().detach().numpy())

            # fig.add_subplot(1, col, 7)
            # plt.title('color')
            # plt.imshow(label[0][1].cpu().detach().numpy())

            # fig.add_subplot(1, col, 8)
            # plt.title('color')
            # plt.imshow(label[0][2].cpu().detach().numpy())

            # fig.add_subplot(1, col, 9)
            # plt.title('color')
            # plt.imshow(label[0][3].cpu().detach().numpy())


            plt.show()

# print(y/len(dataset)*100)
# print('Label : ', Data)
print('Pred  90 : ', pred[0]/Data[0])
print('Pred  -45 : ', pred[1]/Data[1])
print('Pred  0 : ', pred[2]/Data[2])
print('Pred  45 : ', pred[3]/Data[3])