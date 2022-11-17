import cv2
import numpy as np
import matplotlib.pyplot as plt

print('begin')

path =  r'C:\Desktop\Desktop Summary\EEE\Ballinmaze\maze2.jpg'
#vid = cv2.VideoCapture(1)

maze = cv2.imread(path)
original = maze.copy() #save a copy of the original raw image we get to make a masked image afterwards as the image can be changed after threshold()
mazegray = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)#Turn the image into grayscale

#edged= cv2.Canny(mazegray,30,200)

ret,thresh = cv2.threshold(mazegray, 127, 255, 0) #Turn the image into binary scale
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# Find contours, RETR_TREE means that every contour in the image
print("Number of Contours found = " + str(len(contours)))
mask = np.zeros(maze.shape, dtype=np.uint8)
maxarea=0
largest= None

# To choose the largest contours, we can certainly get the contour of the board
for cnt in contours:
    x1,y1 = cnt[0][0]
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True) #approxPolyDP(input_curve, epsilon, closed), used to approximate the shape of the contour
                                                                        #where input_curve represents the input polygon whose contour must be approximated with specified precision,
                                                                        #epsilon represents the maximum distance between the approximation of a shape contour of the input polygon and the original input polygon
                                                                        #closed is a Boolean value whose value is true if the approximated curve is closed or the value is false if the approximated curve is not closed.
    area =cv2.contourArea(cnt)
    if len(approx) == 4 and area > maxarea: #Find the largest contour with 4 edges (rectangle)
        maxarea = area
        largest = cnt

largestarea =cv2.contourArea(largest) 
x, y, w, h = cv2.boundingRect(largest)

print('Area of contour is:',largestarea)
print(x,y,w,h) #X and Y coordinate, Width and height, use to crop out maze from image.

cv2.drawContours(maze, [largest], 0, (0,255,0), 3) # draw the largest contour on the picture
cv2.drawContours(mask,[largest], 0, (255,255,255), -1) 
mask = cv2.bitwise_and(mask, original) # Make a AND GATE process, only leave the white area and wipe out all black area, here the white area is the contour 

#cv2.rectangle(mask, (x, y), (x + w, y + h), (255,255,255), 2) #sketch the edge of the mask, can be ignored
roi = mask[y:y + h, x:x + w] #region of image, extract the smallest square area that surrounding the contour, so that we can have a almost consistent maze
mask_resized= cv2.resize(roi,(800,800)) #resize it to 800 x 800
maze_resized= cv2.resize(maze,(800,800)) #resize it to 800 x 800

while True:
    cv2.imshow('thresh', thresh)
    cv2.imshow('Contours', maze_resized)
    cv2.imshow('maze', mask_resized)


    key = cv2.waitKey(1)

    #if key == ord('s'):
        

    if key == ord('q'): #press q (no Caplock) to quit and end the program

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
