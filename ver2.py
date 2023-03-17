from collections import deque
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import time,os
# import serial
# import sys
from datetime import datetime
from time import strftime
import multiprocessing as mp

# from time import sleep

# ser=serial.Serial()
# ser.baudrate =115200
# ser.port='COM4'
# ser.open()
# test_packets=[]

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points 
test_packet=[]
greenLower = (50, 86, 6)
greenUpper = (90, 255, 255)
count=0

def serial_IO(test_packet):
	serialout=bytearray(test_packet)
	print(serialout)
	# ser.write(test_packet)
			# while (ser.in_waiting > 0):
			# 	print(ser.read_until().decode("utf-8"), end = '') 
			# print("")

#looping
def ball_track():

	# greenLower = (50, 86, 6)
	# greenUpper = (90, 255, 255)
	count=0
	centers=[]
	arrow=0
	arrowcount=0
	a=0
	b=0
	check=0

	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		if cv2.contourArea(c) > 25 and count<100:
			count=0
			x,y,w,h =cv2.boundingRect(c)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
			center=(int(x+w/2),int(y+h/2))
			xcor=int(x+w/2)
			ycor=int(y+h/2)
			print(center)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			centers.append(center)
			arrowcount=arrowcount+1
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

		
		elif cv2.contourArea(c) < 25 :
			center="ball disappeared"
			test_packet=[]		
			return test_packet
		
		
def ball_failure():
	print("ball droped")
	
if __name__ == '__main__':
	vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	count2=0
	while True:
		ret,frame=vid.read()
		frame = imutils.resize(frame, width=255)
		blurred = cv2.GaussianBlur(frame, (5, 5), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)	
		# construct a mask for the color green
		mask = cv2.inRange(hsv, greenLower, greenUpper)
		# find contours in the mask and initialize the current
		cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		ball_track()
		# p1=mp.Process(target=serial_IO,args=(test_packet))
		# p1.start()
		# p1.join()
		cv2.imshow("Frame", frame)
		cv2.imshow("mask",mask)
		cv2.imshow("hsv",hsv)
		
		if len(test_packet)>9:
			count2=0
			serial_IO(test_packet)
		if len(test_packet)<9:
			count2 += 1

		if count2 > 199:
			ball_failure()
			break

		key = cv2.waitKey(1) & 0xFF
		 		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

