from collections import deque
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points 
greenLower = (50, 86, 6)
greenUpper = (90, 255, 255)

vid = cv2.VideoCapture(0)
#looping
while True:
	# grab the current frame
	ret,frame = vid.read()
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=250)
	blurred = cv2.GaussianBlur(frame, (5, 5), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color green
	mask = cv2.inRange(hsv, greenLower, greenUpper)
                                                                                                                                                                                                                                                                                                                                                                                                                    
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		if cv2.contourArea(c) > 50:
			x,y,w,h =cv2.boundingRect(c)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
			center=(int(x+w/2),int(y+h/2))
			print(center)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	cv2.imshow("Frame", frame)
	cv2.imshow("mask",mask)
	cv2.imshow("hsv",hsv)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break


# close all windows
cv2.destroyAllWindows()
