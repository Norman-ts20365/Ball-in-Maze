import cv2
import numpy as np
import matplotlib.pyplot as plt

print('begin')

path =  r'C:\Desktop\Desktop Summary\EEE\Ballinmaze\maze.jpg'
#vid = cv2.VideoCapture(1)

maze = cv2.imread(path)
original = maze.copy()
mazegray = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)

#edged= cv2.Canny(mazegray,30,200)

ret,thresh = cv2.threshold(mazegray, 127, 255, 0) 
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours found = " + str(len(contours)))
mask = np.zeros(maze.shape, dtype=np.uint8)
for cnt in contours:
    x1,y1 = cnt[0][0]
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    area =cv2.contourArea(cnt)
    
    if len(approx) == 4 and area >200000:
      print(area)
      x, y, w, h = cv2.boundingRect(cnt)
      print(x,y,w,h)
      cv2.drawContours(maze, [cnt], 0, (0,255,0), 3)
      cv2.drawContours(mask,[cnt], 0, (255,255,255), -1)
      

mask = cv2.bitwise_and(mask, original)



mask_resized= cv2.resize(mask,(1000,1000))
maze_resized= cv2.resize(maze,(1000,1000))

while True:
    cv2.imshow('thresh', thresh)
    cv2.imshow('Contours', maze_resized)
    cv2.imshow('mask', mask_resized)

    key = cv2.waitKey(1)

    #if key == ord('s'):
        

    if key == ord('q'):

        cv2.destroyAllWindows()
        break




#############Videocapture part as reference###########

#while True:
    #ret, frame = vid.read()
    #cv2.imshow('frame', frame)
    #key = cv2.waitKey(1)
    #if key == ord('s'):

#grey_savedpic = cv2.imread(path + 'maze.jpg',cv2.IMREAD_GRAYSCALE)
#saved_grey = cv2.imwrite(path + 'savedgrey.jpg',grey_savedpic)
#ret,thresh = cv2.threshold(grey_savedpic,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#img,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #elif key == ord('q'):
        #vid.release()
        #cv2.destroyAllWindows()
        #break

   
print('end')
