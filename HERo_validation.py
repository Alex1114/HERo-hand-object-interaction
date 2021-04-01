import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.draw import polygon
from skimage.feature import peak_local_max
import torch.nn.functional as F
import json
import os

from scipy import ndimage
import scipy.misc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets ,transforms
import torchvision
from matplotlib import cm
import os
import copy
import sys
import matplotlib.image as mpimg

class HERo(nn.Module):
    def __init__(self, n_classes):
        super(HERo, self).__init__()
        self.color_trunk = torchvision.models.resnet101(pretrained=True)
        del self.color_trunk.fc, self.color_trunk.avgpool, self.color_trunk.layer4
        self.depth_trunk = copy.deepcopy(self.color_trunk)
        self.conv1 = nn.Conv2d(2048, 512, 1)
        self.conv2 = nn.Conv2d(512, 128, 1)
        self.conv3 = nn.Conv2d(128, n_classes, 1)
    def forward(self, color, depth):
        # Color
        color_feat_1 = self.color_trunk.conv1(color) # 3 -> 64
        color_feat_1 = self.color_trunk.bn1(color_feat_1)
        color_feat_1 = self.color_trunk.relu(color_feat_1)
        color_feat_1 = self.color_trunk.maxpool(color_feat_1) 
        color_feat_2 = self.color_trunk.layer1(color_feat_1) # 64 -> 256
        color_feat_3 = self.color_trunk.layer2(color_feat_2) # 256 -> 512
        color_feat_4 = self.color_trunk.layer3(color_feat_3) # 512 -> 1024
        # Depth
        depth_feat_1 = self.depth_trunk.conv1(depth) # 3 -> 64
        depth_feat_1 = self.depth_trunk.bn1(depth_feat_1)
        depth_feat_1 = self.depth_trunk.relu(depth_feat_1)
        depth_feat_1 = self.depth_trunk.maxpool(depth_feat_1) 
        depth_feat_2 = self.depth_trunk.layer1(depth_feat_1) # 64 -> 256
        depth_feat_3 = self.depth_trunk.layer2(depth_feat_2) # 256 -> 512
        depth_feat_4 = self.depth_trunk.layer3(depth_feat_3) # 512 -> 1024
        # Concatenate
        feat = torch.cat([color_feat_4, depth_feat_4], dim=1) # 2048
        feat_1 = self.conv1(feat)
        feat_2 = self.conv2(feat_1)
        feat_3 = self.conv3(feat_2)
        return nn.Upsample(scale_factor=2, mode="bilinear")(feat_3)

net = HERo(2)
net.load_state_dict(torch.load('/home/arg-medical/HERo-hand-object-interaction/models/2class_net_108.pth'))
net = net.cuda().eval()

name = []
f = open('/home/arg-medical/HERo-hand-object-interaction/dataset/test.txt', "r")
for i, line in enumerate(f):
    name.append(line.replace("\n", ""))

image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])

angle_list = []
for i in range(16):
    angle_list.append(i*22.5)

def img_tensor(idx_name):
    color_o = cv2.imread('/home/arg-medical/HERo-hand-object-interaction/dataset/'+"color/color"+idx_name)
    color_o = color_o[:,:,[2,1,0]]
    depth_o = cv2.imread('/home/arg-medical/HERo-hand-object-interaction/dataset/'+"depth/depth"+idx_name, 0)
    label = cv2.imread('/home/arg-medical/HERo-hand-object-interaction/dataset/'+"label/label"+idx_name)
    center = (color_o.shape[0]/2, color_o.shape[1]/2)
    Sample_angle = []
    
    for angle in angle_list:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        color_img = cv2.warpAffine(color_o, M, (color_o.shape[0], color_o.shape[1]))
        depth_img = cv2.warpAffine(depth_o, M, (color_o.shape[0], color_o.shape[1]))

        # uint8 -> float
        color = (color_img/255.).astype(float)
        # BGR -> RGB and normalize
        color_rgb = np.zeros(color.shape)
        
        for i in range(3):
            color_rgb[:, :, i] = (color[:, :, 2-i]-image_net_mean[i])/image_net_std[i]
            depth = (depth_img/1000.).astype(float) # to meters
            
            # SR300 depth range
            depth = np.clip(depth, 0.0, 1.2)
  
            # Duplicate channel and normalize
            depth_3c = np.zeros(color.shape)
    
        for i in range(3):
            depth_3c[:, :, i] = (depth[:, :]-image_net_mean[i])/image_net_std[i]
            # Unlabeled -> 2; unsuctionable -> 0; suctionable -> 1
            transform = transforms.Compose([transforms.ToTensor(),])
            color_tensor = transform(color_rgb).float()
            depth_tensor = transform(depth_3c).float()
    
            sample = {"color": color_tensor, "depth": depth_tensor, "origin_color": color_img, "origin_depth": depth_img, "label": label}
            Sample_angle.append(sample)
            
    return Sample_angle

def pred_visualize(sample, name, angle, Net):
    
    # read rgb-d and depth image from dataloader
    color_tensor = sample['color'].cuda()
    depth_tensor = sample['depth'].cuda()
    origin_color = sample['origin_color']
    origin_depth = sample['origin_depth']
    c, h, w = color_tensor.shape
    color_tensor = color_tensor.reshape(1, c, h, w)
    depth_tensor = depth_tensor.reshape(1, c, h, w)

    # read original image (heightmap)
    heightmap_c = origin_color
    heightmap_c = cv2.resize(heightmap_c, (h, w))
    heightmap_d = origin_depth
    heightmap_d = cv2.resize(heightmap_d, (h, w))

    # make prediction
    predict = Net.forward(color_tensor, depth_tensor)


    # process prediction result
    graspable = predict[0, 1].detach().cpu().numpy()
    graspable = cv2.resize(graspable, (h, w))
    graspable[heightmap_d==0] = 0

    graspable[graspable>=1] = 0.99999
    graspable[graspable<0] = 0

    graspable = cv2.GaussianBlur(graspable, (7, 7), 0)
    affordanceMap = (graspable/np.max(graspable)*255).astype(np.uint8)
    affordanceMap = cv2.applyColorMap(affordanceMap, cv2.COLORMAP_JET)
    affordanceMap = affordanceMap[:,:,[2,1,0]]

    # combine prediction result (heatmap) and original image (heightmap color image)
    Result = heightmap_c[:,:,[2,1,0]] + affordanceMap
    Result = cv2.addWeighted(heightmap_c, 0.7, affordanceMap, 0.3, 0)

    plt.figure(figsize=(15,15))
    plt.subplot(131)
    plt.title('Color image Angle : '+str(angle))
    plt.imshow(heightmap_c)
    plt.subplot(132)
    plt.title('AffordanceMap')
    plt.imshow(affordanceMap)
    plt.subplot(133)
    plt.title('Result')
    plt.imshow(Result)
    plt.show()

def rotate_pred(data, model):
    angle_split = img_tensor(data)
    i = 0
    for ing in angle_split:
        pred_visualize(ing, data, i*22.5, model)
        i += 1

rotate_pred(name[1], net)