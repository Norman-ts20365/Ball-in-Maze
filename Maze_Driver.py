import numpy as np
import cv2
import matplotlib.pyplot as plt
import serial
import sys
from datetime import datetime
from time import sleep
import multiprocessing
from multiprocessing import Process, Queue, Manager, Value

from Board_Detection import board_detection

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

def main(x,y,t,record,queue):   
    # Responsible for the operation of each learning runtime sequence
    # Must use the training class to generate the correct parameter sets 
    # Must run the timer and send the record trigger and the stop to the ESP32
    # Must evaluate the fitness function and handle saving results to csv
    sleep(5) # need to stop record triggering before it gets to see green
    record.value = 1 # trigger this at the same time as the start packet
    while True:
        return
    
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
    cv2.imshow('Image ', result)   
    return test_packet
          

def image_processor(record, x_tracking, y_tracking, times, queue):
    # Image processor initialisation

    count2 = 0
    test_packet=[]
    YellowLower=(90,50,70)
    YellowUpper=(128,255,255)
    greenLower = (35,43,46)#(35, 43, 46)#(50, 43, 46)
    greenUpper = (77,255,255)#(77, 255, 255)#(90, 255, 255)
    vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    # Image processor loop
    while True:
        detected_board = board_detection(False)

        if detected_board is not None:
         
            hsv2= cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv2, greenLower, greenUpper)
            kernel = np.ones((5,5),np.uint8)
            maskball = cv2.morphologyEx(mask2.copy(), cv2.MORPH_OPEN, kernel)
            maskball2 =cv2.dilate(maskball,kernel,iterations = 1)
            cv2.imshow("mask2",maskball2)
            cnts, hierarchy = cv2.findContours(maskball2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
            test_packet=ball_tracking(result,cnts)
           
            if (record.value == 1) and len(test_packet)>9: # used to log/ plot the response
                xcor=test_packet[2]
                ycor=test_packet[3]
                sec=test_packet[5]
                microsec=test_packet[6]+test_packet[7]+test_packet[8]
                x_tracking.append(xcor)
                y_tracking.append(ycor)
                time = float((sec * 1000) + (float(microsec) / 1000))
                times.append(time) # in ms

            cv2.imshow('Image ', result)
            #cv2.imshow("mask",mask2)
            #cv2.imshow("mask2",maskboard2)
            #cv2.imshow("hsv",hsv)
            key = cv2.waitKey(1) & 0xFF # if removed it has a fit

            if len(test_packet) > 9:
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
    ser.port = 'COM4'
    ser.open()
    while True:
        if not queue.empty():
            cmd = queue.get() # sharing from another process the serial write command
            ser.write(cmd)
            #print("Sent: ", cmd)
        if (ser.in_waiting > 0):
            print(ser.read_until().decode("utf-8"), end = '')
    
    
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
    record = Value('i',0)
    mn = Process(target=main, args=(x,y,t,record,queue,))
    srl = Process(target=serial_read, args=(queue,))
    img = Process(target=image_processor, args=(record,x,y,t,queue,))
    pltter = Process(target=plot, args=(x,y,t,))
    pltter.start()
    mn.start()
    srl.start()
    img.start()

    # User input must reside in the main thread
    while True:
        user = input(": ")
        if user == 'S':
            print("Issuing Start")
            queue.put(start)
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
    mn.join() # waits for main to return 
    img.terminate()
    srl.terminate()
    pltter.terminate()
    plot(x, y, t)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
