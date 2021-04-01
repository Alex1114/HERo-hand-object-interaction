#!/usr/bin/env python3

import os 
import cv2
import copy
import math
import time
import struct
import pickle
import numpy as np
from pycpd import DeformableRegistration
import operator
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ROS
import rospy
import roslib
import rospkg
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from medical_msgs.msg import *
from medical_msgs.srv import *
from scipy.spatial.transform import Rotation

# Torch
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets ,transforms
import torch.nn.functional as F

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

class HERo_Predict():
	def __init__(self):
		self.bridge = CvBridge()
		r = rospkg.RosPack()
		self.path = r.get_path("hero_prediction")

		# Switch
		self.switch = True

		# HERo Model
		self.net = HERo(3) # Model Class
		self.net.load_state_dict(torch.load(os.path.join(self.path, "weight/3class_net_75")))
		self.net = self.net.cuda().eval()
		
		# Publisher
		self.pose_pub = rospy.Publisher("HERo/hand_object_pose", HandObjectPose, queue_size=1)
		self.predict_hero = rospy.Publisher("HERo/affordanceMap", Image, queue_size = 1)
		self.affordance_pub = rospy.Publisher("HERo/affordance", Image, queue_size = 1)

		# Subscriber
		rospy.Subscriber("HERo/affordance", Image, self.callback_vis)

		# Mssage filter 
		depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
		image_sub = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
		
		ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 5, 5)
		ts.registerCallback(self.callback)

		# service
		self.predict_switch = rospy.Service("~predict_switch_server", model_switch, self.switch_callback)

		# Camera info
		info = rospy.wait_for_message('camera/color/camera_info', CameraInfo)
		self.fx = info.P[0]
		self.fy = info.P[5]
		self.cx = info.P[2]
		self.cy = info.P[6]

	def callback_vis(self, affordance):

		self.affordance = self.bridge.imgmsg_to_cv2(affordance, "bgr8")
		Result = cv2.addWeighted(self.cv_image, 0.7, self.affordance, 0.3, 0)
		self.predict_hero.publish(self.bridge.cv2_to_imgmsg(Result, "bgr8"))

	def callback(self, rgb, depth):
		if not self.switch:
			return
		
		# Ros image to cv2
		try:
			self.cv_image = self.bridge.compressed_imgmsg_to_cv2(rgb, "bgr8")
			self.cv_depth = self.bridge.imgmsg_to_cv2(depth, "16UC1")
			self.cv_depth_grasp = self.cv_depth.copy()
		except CvBridgeError as e:
			print(e)
		
		# Define msgs
		pose_msgs = HandObjectPose()

		# Processing
		self.color = self.cv_image[:,:,[2,1,0]]
		self.color = (self.color/255.).astype(float)
		self.color_rgb = np.zeros(self.color.shape)
		image_net_mean = np.array([0.485, 0.456, 0.406])
		image_net_std  = np.array([0.229, 0.224, 0.225])

		for i in range(3):
			self.color_rgb[:, :, i] = (self.color[:, :, 2-i]-image_net_mean[i])/image_net_std[i]

		self.cv_depth = np.round((self.cv_depth/np.max(self.cv_depth))*255).astype('int').reshape(1,self.cv_depth.shape[0],self.cv_depth.shape[1])
		self.cv_depth[self.cv_depth > 100] = 0

		self.depth = (self.cv_depth/1000.).astype(float) # to meters
		self.depth = np.clip(self.depth, 0.0, 1.2)
		self.depth_3c = np.zeros(self.color.shape)
		for i in range(3):
			self.depth_3c[:, :, i] = (self.depth[:, :]-image_net_mean[i])/image_net_std[i]
		transform = transforms.Compose([
						transforms.ToTensor(),
					])

		object_angle = self.find_angle()
		
		h, w = self.color_rgb.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, int(object_angle), 1)
		color_rotated = cv2.warpAffine(self.color_rgb, M, (w, h))
		depth_rotated = cv2.warpAffine(self.depth_3c, M, (w, h))

		color_tensor = transform(color_rotated).float()
		depth_tensor = transform(depth_rotated).float()

		color_tensor = color_tensor.cuda()
		depth_tensor = depth_tensor.cuda()
		c, h, w = color_tensor.shape
		color_tensor = color_tensor.reshape(1, c, h, w)
		depth_tensor = depth_tensor.reshape(1, c, h, w)
		with torch.no_grad():
			predict = self.net.forward(color_tensor, depth_tensor)

		# Process prediction result
		graspable = predict[0, 1].detach().cpu().numpy()
		graspable = cv2.resize(graspable, (w, h))
		heightmap_d = cv2.resize(self.cv_depth[0], (w, h))
		graspable[heightmap_d==0] = 0

		graspable[graspable>=1] = 0.99999
		graspable[graspable<0] = 0
		
		graspable = cv2.GaussianBlur(graspable, (7, 7), 0)
		affordanceMap = (graspable/np.max(graspable)*255).astype(np.uint8)
		affordanceMap = cv2.applyColorMap(affordanceMap, cv2.COLORMAP_JET)
		affordanceMap = affordanceMap[:,:,[2,1,0]]

		M = cv2.getRotationMatrix2D(center, -int(object_angle), 1)
		affordanceMap_roation = cv2.warpAffine(affordanceMap, M, (w, h))

		self.affordance_pub.publish(self.bridge.cv2_to_imgmsg(affordanceMap_roation, "bgr8"))
		
		# ================= Get Gripping Point ================== 
		gray = cv2.cvtColor(affordanceMap_roation, cv2.COLOR_RGB2GRAY)
		blurred = cv2.GaussianBlur(gray, (11, 11), 0)
		binaryIMG = cv2.Canny(blurred, 20, 160)
		binary, contours, hierarchy = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		i = 0
		point_x = 0
		point_y = 0
		cX = 0
		cY = 0
		x = 0
		y = 0
		z = 0
		
		for c in contours:
			M = cv2.moments(c)
			if(M["m00"]!=0): 
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				zc = self.cv_depth_grasp[cY, cX]/1000
				if 0 < zc < 0.65:
					i += 1
					point_x += cX
					point_y += cY

		if i > 0:
			x = int(point_x / i)
			y = int(point_y / i)
			z = self.cv_depth_grasp[y, x]/1000
			grasp_point_x, grasp_point_y, grasp_point_z = self.getXYZ(x , y, z)

			pose_msgs.pose.position.x = grasp_point_x
			pose_msgs.pose.position.y = grasp_point_y
			pose_msgs.pose.position.z = grasp_point_z

		# ================= Angle ==================

		rot = Rotation.from_euler('xyz', [object_angle, 0, 0], degrees=True)

		# Convert to quaternions and print
		rot_quat = rot.as_quat()

		# Add to pose msgs				
		pose_msgs.pose.orientation.x = rot_quat[0]
		pose_msgs.pose.orientation.y = rot_quat[1]
		pose_msgs.pose.orientation.z = rot_quat[2]
		pose_msgs.pose.orientation.w = rot_quat[3]

        # ================= Pub ==================

		if pose_msgs.pose.position.x != 0:
			self.pose_pub.publish(pose_msgs)
			print(pose_msgs)
			print("Angle: ", object_angle)
			print("==========================")


		cv_image = None
		cv_depth = None		


	def getXYZ(self, x, y, zc):
		
		x = float(x)
		y = float(y)
		zc = float(zc)
		inv_fx = 1.0/self.fx
		inv_fy = 1.0/self.fy
		x = (x - self.cx) * zc * inv_fx
		y = (y - self.cy) * zc * inv_fy 
		z = zc 

		return z, -1*x, -1*y

	def find_angle(self):

		transform = transforms.Compose([
						transforms.ToTensor(),
					])

		# Roation
		h, w = self.color_rgb.shape[:2]
		center = (w // 2, h // 2)
		roation_angle = []
		angle = [-60, -45, -30, 90, 60, 45, 30, 0]

		for i in angle:
			M = cv2.getRotationMatrix2D(center, int(i), 1)
			color_rotated = cv2.warpAffine(self.color_rgb, M, (w, h))
			depth_rotated = cv2.warpAffine(self.depth_3c, M, (w, h))

			color_tensor = transform(color_rotated).float()
			depth_tensor = transform(depth_rotated).float()

			color_tensor = color_tensor.cuda()
			depth_tensor = depth_tensor.cuda()
			c, h, w = color_tensor.shape
			color_tensor = color_tensor.reshape(1, c, h, w)
			depth_tensor = depth_tensor.reshape(1, c, h, w)
			predict = self.net.forward(color_tensor, depth_tensor)

			# Process prediction result
			graspable = predict[0, 1].detach().cpu().numpy()
			graspable = cv2.resize(graspable, (w, h))
			heightmap_d = cv2.resize(self.cv_depth[0], (w, h))
			graspable[heightmap_d==0] = 0

			graspable[graspable>=1] = 0.99999
			graspable[graspable<0] = 0
			
			graspable = cv2.GaussianBlur(graspable, (7, 7), 0)
			affordanceMap = (graspable/np.max(graspable)*255).astype(np.uint8)
			affordanceMap = cv2.applyColorMap(affordanceMap, cv2.COLORMAP_JET)
			
			affordanceMap = affordanceMap[:,:,[2,1,0]]

			self.cv_image = cv2.resize(self.cv_image, (w, h))
			Result = self.cv_image[:,:,[2,1,0]] + affordanceMap
			Result = cv2.addWeighted(self.cv_image, 0.7, affordanceMap, 0.3, 0)
			
			# OpenCV Processing
			gray = cv2.cvtColor(affordanceMap, cv2.COLOR_RGB2GRAY)
			blurred = cv2.GaussianBlur(gray, (11, 11), 0)
			binaryIMG = cv2.Canny(blurred, 20, 160)
			binary, contours, hierarchy = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			area = 0
			for c in contours:
				area += cv2.contourArea(c)

			roation_angle.append(int(area))
		
		object_angle = angle[roation_angle.index(max(roation_angle))]

		return object_angle

	def switch_callback(self, req):
		resp = model_switchResponse()
		self.switch = req.data
		s = "True" if req.data else "False"
		resp.result = "Switch turn to {}".format(req.data)
		return resp
		
	def onShutdown(self):
		rospy.loginfo("Shutdown.")	

if __name__ == '__main__':
	rospy.init_node('HERo_Predict')
	foo = HERo_Predict()
	rospy.on_shutdown(foo.onShutdown)
	rospy.spin()
