import cv2
import numpy as np

def flatten_board(cnts2,frame):
    new_contours=[] 
    for contour in cnts2:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area=cv2.contourArea(contour)
        if area >= 50:
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
        result = None
        return result


def board_detection():
# Image processor initialisation
    
    blue_lower = (90,50,70)
    blue_upper = (128,255,255) 

    counter = 0
    vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # Image processor loop
    while True:
        ret,frame=vid.read()
        frame=cv2.resize(frame,(765,635))
        cv2.imshow("frame",frame)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  

        # construct a mask for the color green
        maskboard=cv2.inRange(hsv,blue_lower,blue_upper)
        kernel = np.ones((5,5),np.uint8)
        maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
        maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)
        cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        detected_board = flatten_board(cnts2,frame)
        counter += 1

        if counter>20:
            return detected_board
