#!/usr/bin/env python

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
from time import sleep
from multiprocessing import Process
from random import randint

# Packet declaration
jog_Yplus = bytearray([2,7,119,3]) # 2,7,W,3
jog_Yminus = bytearray([2,7,115,3])
jog_Xplus = bytearray([2,7,100,3])
jog_Xminus = bytearray([2,7,97,3])
route = bytearray([2,5,4,0,25,25,125,25,125,125,25,125,3])


# Serial object
ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM5'
ser.open()
sleep(1)

def main():    
    
    while True:
        key = input(": ")
    
        if (key == "R"):
            ser.write(route)
            sleep(1)
            serial_read()
            continue
            
        if (key == "I"):
            image_processor()
            serial_read()
            sleep(0.1)
        
        if (key == "a"):
            ser.write(jog_Xminus)
            sleep(0.1)
            continue

        if (key == "d"):
            ser.write(jog_Xplus)
            sleep(0.1)
            continue
        
        if (key == "w"):
            ser.write(jog_Yplus)
            sleep(0.1)
            continue

        if (key == "s"):
            ser.write(jog_Yminus)
            sleep(0.1)
            continue
            
        if (key == "T"):
            while True: 
                min=datetime.now().strftime("%M")
                min=int(min)
                sec=datetime.now().strftime("%S")
                sec=int(sec)
                microsec=datetime.now().strftime("%f")
                microsec1=int(str(microsec)[:2])
                microsec2=int(str(microsec)[2:4])
                microsec3=int(str(microsec)[4:])
                test_packet=bytearray([2,4,randint(0,50),randint(0,50),min,sec,microsec1,microsec2,microsec3,3])
                print("Sending")
                ser.write(test_packet)
                sleep(0.05)
                while (ser.in_waiting > 0):
                    print(ser.read_until().decode("utf-8"), end = '') 
        
    
        elif (input(": ") == "E"):
            exit()
            break

def exit():
    cv2.destroyAllWindows()
    return sys.exit()
    
def image_processor():
    # Image processor initialisation
    greenLower = (50, 43, 46)
    greenUpper = (90, 255, 255)
    count = 0
    centers = []
    arrow = 0
    arrowcount = 0
    a = 0
    b = 0
    check = 0
    vid = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    count2=0
    # Image processor loop
    while True:
        test_packet=[]
        ret,frame=vid.read()
        frame = imutils.resize(frame, width=255)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)	
		# construct a mask for the color green
        mask = cv2.inRange(hsv, greenLower, greenUpper)
		# find contours in the mask and initialize the current
        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 25 and count<100:
                count=0
                x,y,w,h =cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                center=(int(x+w/2),int(y+h/2))
                xcor=int(x+w/2)
                ycor=int(y+h/2)
                #print(center)
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
                test_packet=([2,4,xcor,ycor,min,sec,microsec1,microsec2,microsec3,3])

		
            elif cv2.contourArea(c) < 25 :
                center="Ball Disappeared"
                test_packet=[]		
		# p1=mp.Process(target=serial_IO,args=(test_packet))
		# p1.start()
		# p1.join()
        cv2.imshow("Frame", frame)
        cv2.imshow("mask",mask)
        cv2.imshow("hsv",hsv)
        
        key = cv2.waitKey(1) & 0xFF
        if len(test_packet) > 9:
            count2=0
            serialout=bytearray(test_packet)
            #print(serialout)
            ser.write(test_packet)
            
        if len(test_packet)<9:
            count2 += 1
            
        if count2 > 199:
            print("FAIL - ball dropped")
            return

        # if the 'q' key is pressed, stop the loop
        if key == ord("E"):
            return
          
        serial_read()
            
            
            
def serial_read():
    while True:
        sleep(0.01)
        while (ser.in_waiting > 0):
            print(ser.read_until().decode("utf-8"), end = '') 
        return
    

if __name__ == "__main__":
    main()
    
