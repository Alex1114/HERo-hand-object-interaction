import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.draw import polygon
from skimage.feature import peak_local_max
import torch.nn.functional as F
import json
import os
from random import sample


# file path , contain 'color' , 'depth' , 'label' , 'json' folders
path = '/dataset'
File = os.listdir(os.getcwd() + path + "/json")
File.sort()

# create data name list
name = os.listdir(os.getcwd() + path + "/color")
name_list = []
for num in name:
		name_list.append(num.split('_')[1].split('.')[0])

# draw label
for name in File:
		label = np.zeros((256,256,3))
		with open(os.getcwd() + path + "/json" + "/" + name,"r") as f:
				data = json.load(f)
				
		for i in range(len(data['shapes'])):
				coord = data['shapes'][i]['points']
				if data['shapes'][i]['label'] == 'good':
						cv2.line(label, (int(coord[0][0]), int(coord[0][1])), (int(coord[1][0]), int(coord[1][1])), (0,255,0),2)
				else:
						cv2.line(label, (int(coord[0][0]), int(coord[0][1])), (int(coord[1][0]), int(coord[1][1])), (255,0,0),2)

				cv2.imwrite(os.getcwd() + path + "/label/label_" + name.split('.')[0].split('_')[1] + ".jpg", label[:,:,[2,1,0]])
# flip 3 times
for idx in name_list:
		color = cv2.imread(os.getcwd() + path + "/color/color_" + idx + ".jpg")
		depth = cv2.imread(os.getcwd() + path + "/depth/depth_" + idx + ".jpg")
		label = cv2.imread(os.getcwd() + path + "/label/label_" + idx + ".jpg")
		
		for n in range(-1,2):
				color_ = cv2.flip(color,n)
				depth_ = cv2.flip(depth,n)
				label_ = cv2.flip(label,n)
				cv2.imwrite(os.getcwd() + path + "/color/color_" + idx + "_" + str(n+1) + ".jpg", color_)
				cv2.imwrite(os.getcwd() + path + "/depth/depth_" + idx + "_" + str(n+1) + ".jpg", depth_)
				cv2.imwrite(os.getcwd() + path + "/label/label_" + idx + "_" + str(n+1) + ".jpg", label_)


# Create training & testing list
image_file = os.listdir(os.getcwd() + path + "/color")
image_file.sort()

data_list = []
for name in image_file:
    data_list.append(name.split("color")[1])

test = sample(data_list, 10)

train = list(set(data_list).difference(set(test)))

f = open(os.getcwd() + path + "/test.txt", "a")
for idx in test:
    f.write( idx + "\n")
f.close()

f = open(os.getcwd() + path + "/train.txt", "a")
for idx in train:
    f.write(idx + "\n")
f.close()