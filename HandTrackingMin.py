import cv2
import mediapipe as mp
import time

from cv2 import COLOR_BGR2RGB

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands =mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

prev_time = 0
cur_time = 0

while True:
	success, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)
	
	if results.multi_hand_landmarks:
		for hand_lms in results.multi_hand_landmarks:
			for id, lm in enumerate(hand_lms.landmark):
				# print(id, lm)
				height, width, chen = img.shape
				cx, cy = int(lm.x * width), int(lm.y * height)
				print(f"id {id}: ", cx, cy)
				if id == 0:
					cv2.circle(img, (cx, cy), 25, (255,100,130), cv2.FILLED)
				
			mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
	
	cur_time = time.time()
	fps = 1/(cur_time-prev_time)
	prev_time = cur_time
	
	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
	(255,0,255), 3)
	
	cv2.imshow("Image", img)
	cv2.waitKey(1)
	