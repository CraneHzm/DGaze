# Copyright (c) Hu Zhiming jimmyhu@pku.edu.cn 2019/6/20 All Rights Reserved.


#################### Libs ####################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import math


# Calculate the angular distance (visual angle) between 2 angular gaze position.
# gth: the ground truth angular gaze position.
# prd: the predicted angular gaze position.
def CalAngularDist(gth, prd):
	#the parameters of our Hmd (HTC Vive).
	#Vertical FOV.
	VerticalFov = math.pi*110/180;
	#Size of a half screen.
	ScreenWidth = 1080;
	ScreenHeight = 1200;
	ScreenCenterX = 0.5*ScreenWidth
	ScreenCenterY = 0.5*ScreenHeight
	#the pixel distance between eye and the screen center.
	ScreenDist = 0.5* ScreenHeight/math.tan(VerticalFov/2);
	
	#transform the angular coords to screen coords.
	gth = AngularCoord2ScreenCoord(gth);
	prd = AngularCoord2ScreenCoord(prd);
	#transform the screen coords to pixel coords.
	gth[0] = gth[0]*ScreenWidth;
	gth[1] = gth[1]*ScreenHeight;
	prd[0] = prd[0]*ScreenWidth;
	prd[1] = prd[1]*ScreenHeight;
	
	#the distance between eye and gth.
	eye2gth = np.sqrt(np.square(ScreenDist) + np.square(gth[0] - ScreenCenterX) + np.square(gth[1] - ScreenCenterY));
	#the distance between eye and prd.
	eye2prd = np.sqrt(np.square(ScreenDist) + np.square(prd[0] - ScreenCenterX) + np.square(prd[1] - ScreenCenterY));
	#the distance between gth and prd.
	gth2prd = np.sqrt(np.square(prd[0] - gth[0]) + np.square(prd[1] - gth[1]));
	
	#the angular distance between gth and prd.
	angular_dist = 180/math.pi*math.acos((np.square(eye2gth) + np.square(eye2prd) - np.square(gth2prd))/(2*eye2gth*eye2prd));
	return angular_dist
	
def AngularCoord2ScreenCoord(AngularCoord):
	# transform the angular coords ((0 deg, 0 deg) at screen center) to screen coords which are in the range of
	# 0-1. (0, 0) at Bottom-left, (1, 1) at Top-right
	
	# the parameters of our Hmd (HTC Vive).
	# Vertical FOV.
	VerticalFov = math.pi*110/180;
	# Size of a half screen.
	ScreenWidth = 1080;
	ScreenHeight = 1200;
	# the pixel distance between eye and the screen center.
	ScreenDist = 0.5* ScreenHeight/math.tan(VerticalFov/2);
	
	ScreenCoord = np.zeros(2)
	
	# the X coord.
	ScreenCoord[0] = 0.5 + (ScreenDist * math.tan(math.pi*AngularCoord[0] / 180)) / ScreenWidth; 
	# the Y coord.
	ScreenCoord[1] = 0.5 + (ScreenDist * math.tan(math.pi*AngularCoord[1] / 180)) / ScreenHeight;
	return ScreenCoord
	
	
if __name__ == "__main__":
	data = np.loadtxt('test/predictions.txt')
	size = data.shape[0]
	dist = np.zeros(size)
	for i in range(size):
		#print(data[i, :])
		dist[i] = CalAngularDist(data[i,0:2], data[i,2:4])
	
	print(dist)
	print(dist.mean())
	print(dist.std())
	
	
	