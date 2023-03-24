from collections import deque
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import time,os
import serial
import sys
from datetime import datetime
from time import strftime
import threading as th

# define the lower and upper boundaries of the "green"
# ball in the HSV color space
test_packet=[]
YellowLower=(100,50,50)
YellowUpper=(130,255,255) 
greenLower = (50, 86, 46)
greenUpper = (90, 255, 255)
count=0

# serial output part, generate a packet with coordinate, time and send it to serial.
def serial_IO(test_packet):
	serialout=bytearray(test_packet)
	print(serialout)
	# ser.write(test_packet)
			# while (ser.in_waiting > 0):
			# 	print(ser.read_until().decode("utf-8"), end = '') 
			# print("")

#The ball tracking part 
def ball_track(result,cnts):

	count=0
	centers=[]
	arrow=0
	arrowcount=0
	a=0
	b=0
	check=0
#If any contour detected, choose the contour with maximum area which is the green ball
	if len(cnts) > 0: 
		c = max(cnts, key=cv2.contourArea)
		if cv2.contourArea(c) > 5 and count<100: 
			count=0
			x,y,w,h =cv2.boundingRect(c)
			# Draw the rectangle of ball to define position
			cv2.rectangle(result,(x,y),(x+w,y+h),(255,255,0),2)

			# Original center is used to draw the center of the ball on 765x765 picture, while center is used in 255*255 to transfer to serial
			originalcenter=(int((x+w/2)),int((y+h/2)))
			center=(int((x+w/2)/3),int((y+h/2)/3))
			xcor=int((x+w/2)/3)
			ycor=int((y+h/2)/3)
			print(center)
			cv2.circle(result, originalcenter, 5, (0, 0, 255), -1)

			# A list to store the center, can be used to draw the trace
			centers.append(center)
			arrowcount=arrowcount+1
			
			# Part below is to generate a packet
			min=datetime.now().strftime("%M")
			min=int(min)
			sec=datetime.now().strftime("%S")
			sec=int(sec)
			microsec=datetime.now().strftime("%f")
			microsec1=int(str(microsec)[:2])
			microsec2=int(str(microsec)[2:4])
			microsec3=int(str(microsec)[4:])
			global test_packet
			test_packet=([2,4,xcor,ycor,min,sec,microsec1,microsec2,microsec3,3])
			return test_packet

		
		elif cv2.contourArea(c) < 5 :
			center="ball disappeared"
			test_packet=[]		
			return test_packet
		
		
def ball_failure():
	print("ball droped")

def mainpart():
	vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	count2=0
	while True: 
		test_packet=[]
		ret,frame=vid.read()

		# Frame handling, resize a frame 765*765 and convert it to HSV for masking

		frame =  cv2.resize(frame, (765, 765))
		cv2.imshow("Frame", frame)
		blurred = cv2.GaussianBlur(frame, (5, 5), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		# A hsv mask to detect the board
		maskboard=cv2.inRange(hsv,YellowLower,YellowUpper)
		

		kernel = np.ones((5,5),np.uint8)
		
		# Filter out the noise in board detection
		maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
		maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)

		#
		cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		new_contours=[]
		# Convert contours detected into rectangle
		for contour in cnts2:
			approx=cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
			area=cv2.contourArea(contour)
			if area >=50 :
				new_contours.append(contour)#
			
		# Sort the contours in descending area
		new_contours = sorted(new_contours, key=lambda x: cv2.contourArea(x), reverse=True)

		# Extract 4 largest contours which are 4 corners
		if len(new_contours) >= 4:
			new_contours = new_contours[:4]
			sorted_contours = []
		
		# Arrange corners in clockwise
			for i in range(4):
				x, y, w, h = cv2.boundingRect(new_contours[i])
				sorted_contours.append((x + w / 2, y + h / 2))
			sorted_contours = sorted(sorted_contours, key=lambda x: x[1])
			if sorted_contours[0][0] > sorted_contours[1][0]:
				sorted_contours[0], sorted_contours[1] = sorted_contours[1], sorted_contours[0]
			if sorted_contours[2][0] < sorted_contours[3][0]:
				sorted_contours[2], sorted_contours[3] = sorted_contours[3], sorted_contours[2]
			src_pts = np.array(sorted_contours, np.float32)
			dst_pts = np.array([[0, 0], [765, 0], [765, 765], [0, 765]], np.float32)

			# Rescale the image into an level image
			M = cv2.getPerspectiveTransform(src_pts, dst_pts)
			result = cv2.warpPerspective(frame, M, (765,765))

			result2 = cv2.resize(result,(765,725))
			blurred2 = cv2.GaussianBlur(result, (5, 5), 0)
			hsv2= cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
			mask2 = cv2.inRange(hsv2, greenLower, greenUpper)
			cnts, hierarchy = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			# Track the ball
			ball_track(result,cnts)
			cv2.imshow('Image ', result)
			cv2.imshow("mask",mask2)
			cv2.imshow("mask2",maskboard2)
			cv2.imshow("hsv",hsv)
			cv2.imwrite("image.png",result)
		
		if len(test_packet)>9:
			count2=0
			serial_IO(test_packet)

		if len(test_packet)<9:
			count2 += 1

		if count2 > 1999999:
			ball_failure()
			break

		key = cv2.waitKey(1) & 0xFF
		 		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break
	cv2.destroyAllWindows()
	
	return result,cnts

# Multithreading
def multi():
	result,cnts=mainpart()
	maze_thread = th.Thread(target=mainpart)
	ball_thread = th.Thread(target=ball_track,args=(result, cnts))

	maze_thread.start()
	ball_thread.start()

	maze_thread.join()
	maze_thread.join()

if __name__ == '__main__':
	multi()
	
	
