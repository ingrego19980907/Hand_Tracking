import cv2
import time
import os
import HandTrackingModule as htm

width_cam, height_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

folder_path = "Finger_images"
my_list = os.listdir(folder_path)
# print("My list: ", my_list)
overlay_list = []
for im_path in my_list:
	image = cv2.imread(f"{folder_path}/{im_path}")
	overlay_list.append(image)

# print(overlay_list)

prev_time = 0

detector = htm.HandDetector(detection_confidence=0.75)

tips_ids = [4, 8, 12, 16, 20]
while True:
	success, img = cap.read()
	img = detector.find_hand(img)
	lm_list = detector.find_position(img, draw=False)
	# print(lm_list)
	if len(lm_list) != 0:
		fingers = []
		
		# 4 is id end of Thumb finger
		if lm_list[4][1] > lm_list[3][1]:
			fingers.append(1)
		else:
			fingers.append(0)
		
		# Four fingers
		for finger_id in tips_ids[1:5]:
			if lm_list[finger_id][2] < lm_list[finger_id - 2][2]:
				fingers.append(1)
			else:
				fingers.append(0)
		print(fingers)
		
		total_fingers = sum(fingers)
		# print(total_fingers)
		
		height, width, chen = overlay_list[total_fingers].shape
		img[0:height, 0:width] = overlay_list[total_fingers]
		
		cv2.rectangle(img, (20,225), (170,425), (0,255,100), cv2.FILLED)
		cv2.putText(img, str(total_fingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 30, 40), 25)
	
	cur_time = time.time()
	fps = 1 / (cur_time - prev_time)
	prev_time = cur_time
	
	cv2.putText(img, f"FPS: {int(fps)}", (350,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 100, 100), 3)
	
	cv2.imshow('Image', img)
	cv2.waitKey(1)


