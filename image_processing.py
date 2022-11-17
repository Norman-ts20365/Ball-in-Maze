import cv2
import numpy as np
import matplotlib.pyplot as plt

print('begin')

path =  r'C:\Desktop\Desktop Summary\EEE\Ballinmaze\maze.jpg'
#vid = cv2.VideoCapture(1)

maze = cv2.imread(path)
mazegray = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(mazegray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
thresh_resized=cv2.resize(thresh,(500,500))
while True:
    cv2.imshow('maze', thresh_resized)
    key = cv2.waitKey(1)

    #if key == ord('s'):
        

    if key == ord('q'):
        maze.release()
        cv2.destroyAllWindows()
        break
