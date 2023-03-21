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
YellowLower=(100,50,50)
YellowUpper=(130,255,255) 
greenLower = (50, 43, 46)
greenUpper = (77, 255, 255)
count=0

def board_detection():
	if len(cnts2) > 3 and cv2.contourArea(cnts2[3])>75:		
		x1,y1,w1,h1 =cv2.boundingRect(cnts2[0])
		x2,y2,w2,h2 =cv2.boundingRect(cnts2[1])
		x3,y3,w3,h3 =cv2.boundingRect(cnts2[2])
		x4,y4,w4,h4 =cv2.boundingRect(cnts2[3])
		
		XCORS=[x1,x2,x3,x4]
		YCORS=[y1,y2,y3,y4]
		sorted(XCORS,reverse=True)
		sorted(YCORS,reverse=True)
		Width=XCORS[0]-XCORS[3]
		Height=YCORS[0]-YCORS[3]
		
		cv2.rectangle(frame,(XCORS[3],YCORS[3]),(XCORS[3]+Width,YCORS[3]+Height),(255,0,0),2)
		# cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
		# cv2.rectangle(frame,(x3,y3),(x3+w3,y3+h3),(255,0,0),2)
		# cv2.rectangle(frame,(x4,y4),(x4+w4,y4+h4),(255,0,0),2)
		# roi=frame[YCORS[3]:YCORS[3]+Height,XCORS[3]:XCORS[3]+Width]
		# boardimage = cv2.imshow("roi",roi)
		# return boardimage
	else:
		print("4 corner not found")
		return None




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
		if cv2.contourArea(c) > 75 and count<100:
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

		
		elif cv2.contourArea(c) < 75 :
			center="ball disappeared"
			test_packet=[]		
			return test_packet
		
		
def ball_failure():
	print("ball droped")

	
if __name__ == '__main__':
	vid = cv2.VideoCapture(1,cv2.CAP_DSHOW)
	count2=0
	while True: 
		test_packet=[]
		ret,frame=vid.read()

		frame = imutils.resize(frame, width=255)
		blurred = cv2.GaussianBlur(frame, (5, 5), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


		maskboard=cv2.inRange(hsv,YellowLower,YellowUpper)
		mask = cv2.inRange(hsv, greenLower, greenUpper)

		kernel = np.ones((5,5),np.uint8)
		
		maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
		maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)

		cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		new_contours=[]
		for contour in cnts2:
			approx=cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
			area=cv2.contourArea(contour)
			if area >=100 :
				new_contours.append(contour)

		new_contours = sorted(new_contours, key=lambda x: cv2.contourArea(x), reverse=True)
		if len(new_contours) >= 4:
			new_contours = new_contours[:4]
			sorted_contours = []

			for i in range(4):
				x, y, w, h = cv2.boundingRect(new_contours[i])
				sorted_contours.append((x + w / 2, y + h / 2))
			sorted_contours = sorted(sorted_contours, key=lambda x: x[1])
			if sorted_contours[0][0] > sorted_contours[1][0]:
				sorted_contours[0], sorted_contours[1] = sorted_contours[1], sorted_contours[0]
			if sorted_contours[2][0] < sorted_contours[3][0]:
				sorted_contours[2], sorted_contours[3] = sorted_contours[3], sorted_contours[2]
			src_pts = np.array(sorted_contours, np.float32)
			dst_pts = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], np.float32)
			M = cv2.getPerspectiveTransform(src_pts, dst_pts)
			# contour = new_contours[i]
			# rect = cv2.minAreaRect(contour)
			# box = cv2.boxPoints(rect)
			# box = np.int0(box)
			# cv2.drawContours(frame, [box], 0, (255, 0, 255), 2)
			result = cv2.warpPerspective(frame, M, (255,255))
			cv2.imshow('Image ', result)


		# for contour in cnts2:
		# 	approx=cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		# 	if len(approx) == 4:
		# 		rect = np.float32(approx).reshape(4, 2)
		# 		dst = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], np.float32)
		# 		M = cv2.getPerspectiveTransform(rect, dst)
		# 		warped = cv2.warpPerspective(frame.copy(), M, (255, 255))
		# 		result = warped[0:255, 0:255]
		# 		cv2.imshow("board",result)
				
   
		
		print(cnts2)
		# board_detection()
		ball_track()
		# p1=mp.Process(target=serial_IO,args=(test_packet))
		# p1.start()
		# p1.join()
		cv2.imshow("Frame", frame)
		cv2.imshow("mask",mask)
		cv2.imshow("mask2",maskboard2)
		cv2.imshow("hsv",hsv)
		
		
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

