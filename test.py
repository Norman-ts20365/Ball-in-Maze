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
jog_Xplus = bytearray([2,7,100,3]) # STX, JOG_TYPE, D, ETX
jog_Xminus = bytearray([2,7,97,3])
#route = bytearray([2,5,3,0,100,3,46,3,46,40,3])
route = bytearray([2,5,9,0,125,3,100,3,70,3,44,3,44,35,44,77,63,102,63,138,2,134,3])
start = bytearray([2,6,10,3])
stop = bytearray([2,6,11,3])
home = bytearray([2,6,12,3])
reset = bytearray([2,6,13,3])
fail = bytearray([2,6,14,3])


# Serial object
# ser = serial.Serial()
# ser.baudrate = 115200
# ser.port = 'COM4'
# ser.open()
# sleep(1)

test_packet=[]
YellowLower=(100,50,50)
YellowUpper=(130,255,255) 
greenLower = (70,43,46)#(35, 43, 46)#(50, 43, 46)
greenUpper = (90,255,255)#(77, 255, 255)#(90, 255, 255)
count=0

def main():    
    # import path
    while True:
        serial_read()
        key = input(": ")
    
        if (key == "R"):
            ser.write(route)
            sleep(1)
            serial_read()
            continue
            
        if (key == "I"):
            x,y,t = image_processor()
            plot(x,y,t)
            serial_read()
            sleep(0.1)
            continue
                   
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
            
        if (key == "H"):
            ser.write(home)
            sleep(1)
            continue
    
        elif (input(": ") == "E"):
            exit()
            break
            

def exit():
    cv2.destroyAllWindows()
    return sys.exit()

# It use the raw frame and extract the board with 4 blue corners, then use warping algrorithm to make the board flat to fix the effect of tilting.

def board_detection(cnts2,frame):
    new_contours=[] 
    for contour in cnts2:
        approx=cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area=cv2.contourArea(contour)
        if area >=50 :
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
        dst_pts = np.array([[0, 0], [765, 0], [765, 635], [0, 635]], np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(frame, M, (765,635))
        return result
    else:
        result=frame
        return result

# ball tracking part below, it use the extracted board after board_detection part and seeks for the postion of
# green ball, after displaying the position of ball using a rectangle around it, it generates a variable called
# test_packet including information such as coordinates, time. the variable then is transferred to serial part.
def ball_tracking(result,cnts):   
    count=0
    centers=[]
    arrowcount=0
    test_packet=[]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 15 and count<100:  # contour area was 25
            count=0
            x,y,w,h =cv2.boundingRect(c)
            cv2.rectangle(result,(x,y),(x+w,y+h),(255,255,0),2)
            center=(int((x+w/2)/3),int((y+h/2)/3))
            xcor=int((x+w/2)/3)
            ycor=int((y+h/2)/3)
            #print(center)
                #print("X: ", xcor, "Y: ", ycor)
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            centers.append(center)
            arrowcount=arrowcount+1
            minute=datetime.now().strftime("%M") # PROBLEM HERE - YOU HAD USED A KEYWORD MIN
            minute=int(minute)
            sec=datetime.now().strftime("%S")
            sec=int(sec)
            microsec=datetime.now().strftime("%f")
            microsec1=int(str(microsec)[:2])
            microsec2=int(str(microsec)[2:4])
            microsec3=int(str(microsec)[4:])
            test_packet=([2,4,xcor,ycor,minute,sec,microsec1,microsec2,microsec3,3])
        elif cv2.contourArea(c) < 15 :
            center="ball disappeared"
            
            test_packet=[]          
    cv2.imshow('Image ', result)     
    return test_packet
    
    #cv2.imshow("mask",mask2)
    #cv2.imshow("mask2",maskboard2)
    #cv2.imshow("hsv",hsv)

# The main function of image processing part, it receives frames from camera and blurs it to extract the board and the ball in higher accuracy

def image_processor():
    # Image processor initialisation
    vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    count2=0
    x_tracking = []
    y_tracking = []
    times = []
    record = 0
    pathread=False
    # Image processor loop
    while True:
        
        ret,frame=vid.read()
        frame=cv2.resize(frame,(765,635))
        cv2.imshow("frame",frame)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  
        # construct a mask for the color green
        maskboard=cv2.inRange(hsv,YellowLower,YellowUpper)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        kernel = np.ones((5,5),np.uint8)
        maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
        maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)
        cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        result=board_detection(cnts2,frame)


            # if pathread==False:
            #     global pathgive
            #     pathgive=r'C:\Desktop\Desktop Summary\EEE\groupproject\mid_test.png'
            #     pathread=True

        hsv2= cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv2, greenLower, greenUpper)
        cnts, hierarchy = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        test_packet=ball_tracking(result,cnts)
        
        key = cv2.waitKey(1) & 0xFF
        if len(test_packet) > 9:
            count2=0
            serialout=bytearray(test_packet)
            print(test_packet)
            print(serialout)
            #print(serialout)
            #ser.write(serialout)
            
        if len(test_packet)<9:
            count2 += 1
            
        # if the 'S' key is pressed, stop the loop
        if key == ord("S"):
            # ser.write(start)
            record = 1
            
        if (record == 1): # used to log/ plot the response
            xcor=test_packet[2]
            ycor=test_packet[3]
            sec=test_packet[5]
            microsec=test_packet[6]+test_packet[7]+test_packet[8]
            x_tracking.append(xcor)
            y_tracking.append(ycor)
            time = float((sec * 1000) + (float(microsec) / 1000))
            times.append(time) # in ms
            
        if count2 > 1999:
            print("FAIL - ball dropped")
            return x_tracking, y_tracking, times
            
        # if the 'E' key is pressed, stop the loop
        if key == ord("E"):
            # ser.write(stop)
            return x_tracking, y_tracking, times
        
        # serial_read()
        #sleep(0.1)
            
# with open("path.py") as path:
#     exec(path.read())
# os.system("path.py")

def serial_read():
    while (ser.in_waiting > 0):
        print(ser.read_until().decode("utf-8"), end = '') 
    return
    
def plot(x, y, t):
    X = np.array(x)
    Y = np.array(y)
    T = np.array(t)
    T = T - T[0] # scale the start time to be zero
    fig, ax = plt.subplots(2)
    ax[0].plot(T,X, color = 'red', linewidth = 1.0)
    ax[0].set_title("X Axis")
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_ylabel("PositionS")
    ax[1].plot(T,Y, color = 'blue', linewidth = 1.0)
    ax[1].set_title("Y Axis")
    ax[1].set_xlabel("Time [ms]")
    ax[1].set_ylabel("Position")
    fig.tight_layout()
    plt.show()
    return

    

if __name__ == "__main__":
    image_processor()
