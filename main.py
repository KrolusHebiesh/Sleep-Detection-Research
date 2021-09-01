from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2

def sound_alarm():
	playsound.playsound("alarm.wav")

def EAR(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_STATE = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
time.sleep(1.0)


while True:

	img = vs.read()
	img = imutils.resize(img, width=450)
	grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	decs = detector(grayim, 0)
 
	for dec in decs:

		sh = predictor(grayim, dec)
		sh = face_utils.shape_to_np(sh)

		LE = sh[lStart:lEnd]
		RE = sh[rStart:rEnd]
		LE = EAR(LE)
		RE = EAR(RE)

		ear = (LE + RE) / 2.0

		LEH = cv2.convexHull(LE)
		REH = cv2.convexHull(RE)
		cv2.drawContours(img, [LEH], -1, (0, 255, 0), 1)
		cv2.drawContours(img, [REH], -1, (0, 255, 0), 1)

		if ear < EYE_AR_THRESH:
			COUNTER += 1

			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_STATE:
					ALARM_STATE = True
					t = Thread(target=sound_alarm)
					t.deamon = True
					t.start()
				cv2.putText(img, "You are sleeping", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNTER = 0
			ALARM_STATE = False
 
	cv2.imshow("Result", img)

vs.stop()
