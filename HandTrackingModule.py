import cv2
import mediapipe as mp
import time

from cv2 import COLOR_BGR2RGB


class HandDetector:
	
	def __init__(self, mode=False, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
		self.mode = mode
		self.max_num_hands = max_num_hands
		self.detection_confidence = detection_confidence
		self.tracking_confidence = tracking_confidence
		
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			self.mode, self.max_num_hands, self.detection_confidence, self.tracking_confidence
		)
		self.mp_draw = mp.solutions.drawing_utils

	def find_hand(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(imgRGB)
		if self.results.multi_hand_landmarks:
			for hand_lms in self.results.multi_hand_landmarks:
				if draw:
					self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
				
		return img
	
	def find_position(self, img, hand_num=0, draw=True):
		lm_list = []
		if self.results.multi_hand_landmarks:
			cur_hand = self.results.multi_hand_landmarks[hand_num]
			for id, lm in enumerate(cur_hand.landmark):
				# print(id, lm)
				height, width, chen = img.shape
				cx, cy = int(lm.x * width), int(lm.y * height)
				# print(f"id {id}: ", cx, cy)
				lm_list.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 15, (255, 100, 130), cv2.FILLED)
					
		return lm_list


def main():
	prev_time = 0
	cur_time = 0
	cap = cv2.VideoCapture(0)
	detector = HandDetector()
	
	while True:
		success, img = cap.read()
		
		img = detector.find_hand(img)
		lm_list = detector.find_position(img)
		if len(lm_list) != 0:
			print(lm_list[0])
		cur_time = time.time()
		fps = 1 / (cur_time - prev_time)
		prev_time = cur_time
		
		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
		            (255, 0, 255), 3)
		
		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == "__main__":
	main()