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
import copy
import math
import pandas as pd

import torchvision.transforms as transforms

import cv2
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import numpy as np
import time
import os

image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])

class HERo_dataset(Dataset):
    name = []

    def __init__(self, n_classes, data_dir, phase, scale=1/(6.4)):
        self.data_dir = data_dir
        self.phase = phase
        self.scale = 1/(6.4)
        f = open(self.data_dir+"train.txt", "r")
        for i, line in enumerate(f):
              self.name.append(line.replace("\n", ""))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        idx_name = self.name[idx]
        color_img = cv2.imread(self.data_dir+"color/color"+idx_name)
        color_img = color_img[:,:,[2,1,0]]
        depth_img = cv2.imread(self.data_dir+"depth/depth"+idx_name, 0)

        label_img = cv2.imread(self.data_dir+"label/label"+idx_name, cv2.IMREAD_GRAYSCALE)
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
        label = np.round(label_img/255.*2.).astype(float)
        # Already 40*40
        label = cv2.resize(label, (int(32), int(32)))
        transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
        color_tensor = transform(color_rgb).float()
        depth_tensor = transform(depth_3c).float()
        label_tensor = transform(label).float()
        sample = {"color": color_tensor, "depth": depth_tensor, "label": label_tensor}
        return sample

dataset = HERo_dataset(2, "/home/arg-medical/HERo-hand-object-interaction/dataset/", "training")

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

epochs = 100
save_every = 3
batch_size = 5
class_weight = torch.ones(2)
# class_weight[1] = 0
net = HERo(2)
net = net.cuda()
criterion = nn.CrossEntropyLoss(class_weight).cuda()
optimizer = optim.SGD(net.parameters(), lr = 1e-3, momentum=0.99)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 25, gamma = 0.1)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 8)

loss_l = []

for epoch in range(epochs):
    loss_sum = 0.0
    ts = time.time()
    for i_batch, sampled_batched in enumerate(dataloader):
        print("\r[{:03.2f} %]".format(i_batch/float(len(dataloader))*100.0), end="\r")
        optimizer.zero_grad()
        color = sampled_batched['color'].cuda()
        depth = sampled_batched['depth'].cuda()
        label = sampled_batched['label'].cuda().long()
        predict = net(color, depth)
        loss = criterion(predict.view(len(sampled_batched['color']), 2,32*32), label.view(len(sampled_batched['color']), 32*32))
        loss.backward()
        loss_sum += loss.detach().cpu().numpy()
        optimizer.step()
    scheduler.step()

    if (epoch+1)%save_every==0:
        torch.save(net.state_dict(), "/home/arg-medical/HERo-hand-object-interaction/models/net_{}.pth".format(epoch+1))
    loss_l.append(loss_sum/len(dataloader))

    print("Epoch: {}| Loss: {}| Time elasped: {}".format(epoch+1, loss_l[-1], time.time()-ts))