from collections import deque
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import time,os
from datetime import datetime
from time import strftime

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points 
greenLower = (50, 86, 6)
greenUpper = (90, 255, 255)
count=0
centers=[]
arrow=0
arrowcount=0
a=0
b=0
check=0
vid = cv2.VideoCapture(0)
#looping

while True:
	# grab the current frame
	ret,frame = vid.read()
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=255)
	blurred = cv2.GaussianBlur(frame, (5, 5), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)	

	# construct a mask for the color green
	mask = cv2.inRange(hsv, greenLower, greenUpper)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if check>0 :
		arrow=cv2.arrowedLine(frame, a, b,(0,255,0), 2, 8, 0,0.1)
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		if cv2.contourArea(c) > 75 and count<100:
			count=0
			x,y,w,h =cv2.boundingRect(c)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
			center=(int(x+w/2),int(y+h/2))
			print(center)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			centers.append(center)
			arrowcount=arrowcount+1

			X1=int(x+w/2)
			Y1=int(y+h/2)
			packet=(X1,Y1,)
		
		elif cv2.contourArea(c) < 75 and count<100:
			center="ball disappeared"
			count=count+1
		
		elif count==100:
			print("ball droped")
			break

		if len(centers) % 20 == 0 and len(centers) >19 and arrowcount==len(centers):
			a=centers[arrowcount-20]
			b=centers[arrowcount-1]
			arrowcount=0
			centers=[]
			check=1
			min=datetime.now().strftime("%M")
			sec=datetime.now().strftime("%S")
			microsec=datetime.now().strftime("%f")
			print(min)
			print(sec)
			print(microsec)

			packet=("")

	cv2.imshow("Frame", frame)
	cv2.imshow("mask",mask)
	cv2.imshow("hsv",hsv)

	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break


# close all windows
cv2.destroyAllWindows()
