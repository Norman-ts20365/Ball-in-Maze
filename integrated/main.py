from multiprocessing import Process, Queue, Manager, Value, Condition
import pickle
import serial
from time import sleep

import cv2
import numpy as np
from datetime import datetime

from Board_Detection import flatten_board
from GUI import gui_main, start_timer
# from Route_Detection import route_detection_main


fail = bytearray([2,6,14,3])
start = bytearray([2,6,10,3])
stop = bytearray([2,6,11,3])

def serial_read(queue):
    # Serial object
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM4'
    ser.open()
    while True:
        # print(queue.empty())
        if not queue.empty():
            cmd = queue.get() # sharing from another process the serial write command
            ser.write(cmd)
            # print("Sent: ", cmd)
        if (ser.in_waiting > 0):
            print(ser.read_until().decode("utf-8"), end = '')

def ball_tracking(result,cnts):   
    count=0
    centers=[]
    arrowcount=0
    test_packet=[]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 25 and count<100:  # contour area was 25
            count=0
            x,y,w,h =cv2.boundingRect(c)
            cv2.rectangle(result,(x,y),(x+w,y+h),(255,255,0),2)
            center2=(int((x+w/2)),int((y+h/2)))
            center=(int((x+w/2)/3),int((y+h/2)/3))
            xcor=int((x+w/2)/3)
            ycor=int((y+h/2)/3)
            #print(center)
                #print("X: ", xcor, "Y: ", ycor)
            cv2.circle(result, center2, 5, (0, 0, 255), -1)
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

    count2=0
    test_packet=[]

    YellowLower=(90,50,70)
    YellowUpper=(128,255,255) 
    greenLower = (35,43,46)#(35, 43, 46)#(50, 43, 46)
    greenUpper = (77,255,255)#(77, 255, 255)#(90, 255, 255)

    vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    # Image processor loop
    while True:
        test_packet=[]

        # Process input image
        ret,frame=vid.read()
        frame=cv2.resize(frame,(765,635))
        # cv2.imshow("frame",frame)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  
        # construct a mask for the color green
        maskboard=cv2.inRange(hsv,YellowLower,YellowUpper)
        # cv2.imshow("blue mask",maskboard)
        kernel = np.ones((5,5),np.uint8)
        maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
        maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)
        cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("blue mask2",maskboard2)
        detected_board = flatten_board(cnts2,frame)

        if detected_board is not None:
         
            hsv2= cv2.cvtColor(detected_board, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv2, greenLower, greenUpper)
            kernel = np.ones((5,5),np.uint8)
            maskball = cv2.morphologyEx(mask2.copy(), cv2.MORPH_OPEN, kernel)
            maskball2 =cv2.dilate(maskball,kernel,iterations = 1)
            # cv2.imshow("mask2",maskball2)
            cnts, hierarchy = cv2.findContours(maskball2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
            test_packet=ball_tracking(detected_board,cnts)

            resizedresult=cv2.resize(detected_board,(1148,953))
            window_name = "Real Time"
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 700,335)
            cv2.imshow(window_name, resizedresult)


            key = cv2.waitKey(1) & 0xFF # if removed it has a fit
            if len(test_packet) > 9:
                count2=0
                #print(test_packet)
                serialout=bytearray(test_packet)
                queue.put(serialout)
                
            
            if len(test_packet)<9:
                count2 += 1
            
            if count2 > 1999: # fail condition
                queue.put(stop)
                return       


if __name__ == "__main__":
    # manager = Manager()

    # multiprocessing-related functions
    queue = Queue()
    condition = Condition()

    # declare processes
    process_gui = Process(target=gui_main, args=(condition,))
    # process_Maze_Driver = Process(target=maze_driver_main, args=(condition,queue,))
    process_serial_read = Process(target=serial_read, args=(queue,))
    process_image_processor = Process(target=image_processor, args=(queue,))

    # run processes in parallel
    process_gui.start()
    # process_Maze_Driver.start()
    
    queue.put(stop)

    with condition:
        condition.wait()
    process_serial_read.start()
    sleep(2)
    process_image_processor.start()
    queue.put(start)
    print("I have started timer")
    # start_timer()

    # wait for processes to return 
    process_gui.join()
    process_serial_read.terminate()
    process_image_processor.terminate()
    # process_Maze_Driver.join()



