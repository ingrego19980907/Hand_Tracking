import cv2
import time
import os

from HandTrackingMin import success

width_cam, height_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

while True:
	
	success, img = cap.read()
	cv2.imshow('Image', img)
	cv2.waitKey(1)


