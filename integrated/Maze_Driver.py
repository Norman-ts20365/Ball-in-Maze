import numpy as np
import cv2
import numpy as np
import cv2
from datetime import datetime
import pickle

import serial
import sys

from Board_Detection import flatten_board

fail = bytearray([2,6,14,3])
stop = bytearray([2,6,11,3])


def ball_tracking(board,cnts):   
    count=0
    centers=[]
    arrowcount=0
    test_packet=[]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 25 and count<100:  # contour area was 25
            count=0
            x,y,w,h =cv2.boundingRect(c)
            cv2.rectangle(board,(x,y),(x+w,y+h),(255,255,0),2)
            center2=(int((x+w/2)),int((y+h/2)))
            center=(int((x+w/2)/3),int((y+h/2)/3))
            xcor=int((x+w/2)/3)
            ycor=int((y+h/2)/3)
            #print(center)
                #print("X: ", xcor, "Y: ", ycor)
            cv2.circle(board, center2, 5, (0, 0, 255), -1)
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
        elif cv2.contourArea(c) < 25 :
            center="ball disappeared"
            test_packet=[]          
    # cv2.imshow('Image ', result)   
    return test_packet

def image_processor(queue):
    # Image processor initialisation
    count2 = 0
    test_packet=[]

    YellowLower=(90,50,70)
    YellowUpper=(128,255,255) 
    GreenLower = (35,43,46)#(35, 43, 46)#(50, 43, 46)
    GreenUpper = (77,255,255)#(77, 255, 255)#(90, 255, 255)

    vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    # maze solving loop until maze solved
    while True:
        test_packet=[]

        # Process input image
        ret,frame=vid.read()
        frame=cv2.resize(frame,(765,635))
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color green
        maskboard=cv2.inRange(hsv,YellowLower,YellowUpper)
        kernel = np.ones((5,5),np.uint8)
        maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
        maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)
        cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # get image of the board
        detected_board = flatten_board(cnts2,frame)

        if detected_board is not None:
            hsv2= cv2.cvtColor(detected_board, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv2, GreenLower, GreenUpper)
            kernel = np.ones((5,5),np.uint8)
            maskball = cv2.morphologyEx(mask2.copy(), cv2.MORPH_OPEN, kernel)
            maskball2 =cv2.dilate(maskball,kernel,iterations = 1)
            cnts, hierarchy = cv2.findContours(maskball2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            # generate packet for serial communication
            test_packet = ball_tracking(detected_board,cnts)

            # display real-time footage
            window_name = "Real Time"
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 50,50)
            cv2.imshow(window_name, detected_board)
            
            if len(test_packet) > 9:
                count2=0
                ball_packet = bytearray(test_packet)
                queue.put(ball_packet)
            
            if len(test_packet)<9:
                count2 += 1
            
            if count2 > 1999: # fail condition
                queue.put(fail)
                return

def maze_driver_main(local_condition, local_queue):
    global queue, condition
    condition = local_condition
    queue = local_queue
    with condition:
        condition.wait()
    print("I am maze driver")



    # file = open("coordinates.txt","rb")
    # route_coordinates = pickle.load(file)
    # file.close

    # # setting up a route packet for serial communication
    # payload_length = len(route_coordinates)
    # route_packet = [2,5,(payload_length & 0xff),(payload_length >> 8)&0xff]
    # for i in range(payload_length):
    #     for j in range(2):
    #         route_packet.append(int(route_coordinates[i][j]/3))
    # route_packet.append(3)
    # route_packet = bytearray(route_packet)

    # # setting up multiprocessing
    # queue.put(route_packet)

    # print("I am maze driver and I have put route")
