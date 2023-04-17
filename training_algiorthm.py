#!/usr/bin/env python
'''
PSO Training Implementation for Fuzzy Logic Motion Controller
Oscar Dilley, April 2023
Univeristy of Bristol
Electrical and Electronic Engineering Year 3
'''
import numpy as np
import cv2
from PSO_training import PsoTraining
import matplotlib.pyplot as plt
import serial
import sys
from datetime import datetime
from time import sleep, perf_counter
import multiprocessing
from multiprocessing import Process, Queue, Manager, Value

# Useful Packet Pre-declaration
jog_Yplus = bytearray([2,7,119,3]) # 2,7,W,3
jog_Yminus = bytearray([2,7,115,3])
jog_Xplus = bytearray([2,7,100,3]) # STX, JOG_TYPE, D, ETX
jog_Xminus = bytearray([2,7,97,3])
start = bytearray([2,6,10,3])
stop = bytearray([2,6,11,3])
home = bytearray([2,6,12,3])
reset = bytearray([2,6,13,3])
fail = bytearray([2,6,14,3])

def main(X,Y,T,flag,record,queue):   
    # Responsible for the operation of each learning runtime sequence
    # Must use the training class to generate the correct parameter sets 
    # Must run the timer and send the record trigger and the stop to the ESP32
    # Must evaluate the fitness function and handle saving results to csv
    sleep(2)
    training = PsoTraining()
    print("Running: {}".format(training)) 
    print("Please use jog funtionality to ensure the maze is level before continuing")
    sleep(1)
    params = training.currentParams
    print("Parameters: {}".format(params))
    xStart = training.xStart
    yStart = training.yStart
    xTarget = training.xTarget
    yTarget = training.yTarget
    floats = np.array([params[2], params[3], params[9], params[10]],np.float32)
    training_packet = bytearray([2,9,xTarget,yTarget,params[0], params[1]]) + bytearray(floats[0]) + bytearray(floats[1]) + bytearray([params[4],params[5], params[6], params[7] ,params[8]]) + bytearray(floats[2]) + bytearray(floats[3]) + bytearray([params[11],3])
    queue.put(training_packet)
    print("Parameters uploaded to motion controller.")
    print("Place the ball at start position: {},{}".format(xStart,yStart))
    sleep(2)
    print("When you are happy the ball is in the correct position, use Shift+S to start")
    while (flag.value != 1):
        continue
    print("Issuing Start")
    queue.put(start)
    record.value = 1 # trigger this at the same time as the start packet
    start_time = perf_counter()
    while ((perf_counter() - start_time) < 5):
        continue
    queue.put(stop)
    record.value = 0
    queue.put(reset)
    T = np.array(T)
    X = np.array(X)
    Y = np.array(Y)
    T = T - T[0] # scale the start time to be zero
    xResult = np.sum(abs(xTarget - X)) # the error should be minimised
    yResult = np.sum(abs(yTarget - Y))
    result = ((xResult - yResult) / 2) # fitness outcome is the mean of X and Y behaviour
    if (training.uploadResult(result) == 0):
        print("Training success")
        return
    else:
        print("WARNING - something went wrong, data may not have been saved correctly")
        return

      
def image_processor(record, x_tracking, y_tracking, flag, times, queue):
    # Image processor initialisation
    xcor = 0
    ycor = 0
    count2=0
    test_packet=[]
    YellowLower=(100,50,50)
    YellowUpper=(130,255,255) 
    greenLower = (35, 43, 46)#(50, 43, 46)
    greenUpper = (77, 255, 255)#(90, 255, 255)
    count=0
    vid = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    # Image processor loop
    sleep(3)
    while True:
        test_packet=[]
        ret,frame=vid.read()
        frame=cv2.resize(frame,(765,635))
        cv2.imshow("frame",frame)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  
        maskboard=cv2.inRange(hsv,YellowLower,YellowUpper)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        kernel = np.ones((5,5),np.uint8)
        maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
        maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)
        cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
            result2 = cv2.resize(result,(765,725))
            blurred2 = cv2.GaussianBlur(result, (5, 5), 0)
            hsv2= cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv2, greenLower, greenUpper)
            cnts, hierarchy = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            count=0
            centers=[]
            arrow=0
            arrowcount=0
            a=0
            b=0
            check=0
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 15 and count<100:  # contour area was 25
                    count=0
                    x,y,w,h =cv2.boundingRect(c)
                    cv2.rectangle(result,(x,y),(x+w,y+h),(255,255,0),2)
                    center=(int((x+w/2)/3),int((y+h/2)/3))
                    center2=(int((x+w/2)),int((y+h/2)))
                    str2=str(center)
                    xcor=int((x+w/2)/3)
                    ycor=int((y+h/2)/3)
                    #print(center)
                    if flag == 0: # to help with start position alignment
                        print("X: ", xcor, "Y: ", ycor)
                    cv2.circle(result, center2, 5, (0, 0, 255), -1)
                    cv2.putText(result,str2,center2, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.75,(0,255,255),1 )
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
                    if (record.value == 1): # used to log/ plot the response
                        x_tracking.append(xcor)
                        y_tracking.append(ycor)
                        time = float((sec * 1000) + (float(microsec) / 1000))
                        times.append(time) # in ms
                elif cv2.contourArea(c) < 5 :
                    center="ball disappeared"
                    test_packet=[]		
            cv2.imshow('Image ', result)

        key = cv2.waitKey(1) & 0xFF # if removed it has a fit
        if (len(test_packet) > 9) and (flag == 1):
            count2=0
            #print(test_packet)
            serialout=bytearray(test_packet)
            queue.put(serialout)
            
        if len(test_packet)<9:
            count2 += 1
            
        if count2 > 1999: # fail condition
            queue.put(fail)
            return
        
        if key == ord("E"): # how to abort from the image processing windows
            queue.put(stop)
            return           

def serial_read(queue):
    # Serial object
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM5'
    ser.open()
    while True:
        if not queue.empty():
            cmd = queue.get() # sharing from another process the serial write command
            ser.write(cmd)
            print("Sent: ", cmd)
            continue
        if (ser.in_waiting > 0):
            print("Serial In: ")
            print(ser.read_until().decode("utf-8"), end = '') 
            continue
    
    
def plot(x, y, t):
    # Plotting the ball position in real time
    X = np.array([0])
    Y = np.array([0])
    T = np.array([0])
    fig, ax = plt.subplots(2)
    ax[0].plot(T,X, color = 'red', linewidth = 1.0)
    ax[0].set_title("X Axis")
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_ylabel("PositionS")
    ax[1].plot(T,Y, color = 'blue', linewidth = 1.0)
    ax[1].set_title("Y Axis")
    ax[1].set_xlabel("Time [ms]")
    ax[1].set_ylabel("Position")
    while True:
        X = np.array(x)
        Y = np.array(y)
        T = np.array(t)
        if (len(T) > 1) and (len(T) == len(X)) and (len(T) == len(Y)):
            T = T - T[0] # scale the start time to be zero
            ax[0].clear()
            ax[1].clear()
            ax[0].plot(T,X, color = 'red', linewidth = 1.0)
            ax[1].plot(T,Y, color = 'blue', linewidth = 1.0)
            fig.tight_layout()
            plt.pause(0.0005)

if __name__ == "__main__":
    print("Welcome to the training routine")
    print("This script utilised particle swarm optimisation (PSO) in order to optimise fuzzy logic parameters")
    print("Number of CPU cores available: ", multiprocessing.cpu_count())
    queue = Queue()
    manager = Manager()
    x = manager.list()
    y = manager.list()
    t = manager.list()
    greenFlag = Value('i',0)
    record = Value('i',0)
    mn = Process(target=main, args=(x,y,t,greenFlag,record,queue,))
    srl = Process(target=serial_read, args=(queue,))
    img = Process(target=image_processor, args=(record,x,y,t,greenFlag,queue,))
    pltter = Process(target=plot, args=(x,y,t,))
    pltter.start()
    mn.start()
    srl.start()
    img.start()

    # User input must reside in the main thread
    sleep(5)
    while True:
        user = input(": ")
        if user == 'S':
            greenFlag.value = 1 # Issue the start to the maze
        elif user == 'E':
            print("Exiting")
            break
        elif user == 'w':
            print("Jog y+")
            queue.put(jog_Yplus)
        elif user == 's':
            print("Jog y-")
            queue.put(jog_Yminus)
        elif user == 'd':
            print("Jog x+")
            queue.put(jog_Xplus)
        elif user == 'a':
            print("Jog x-")
            queue.put(jog_Xminus)

    # If the user selects to exit, handle termination of threads and stop the process
    mn.terminate() # waits for main to return 
    img.terminate()
    srl.terminate()
    pltter.terminate()
    #plot(x, y, t)
    cv2.destroyAllWindows()
    sys.exit()

    
