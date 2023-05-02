import cv2
import numpy as np

def flatten_board(cnts2,frame):
    new_contours=[] 
    for contour in cnts2:

        area=cv2.contourArea(contour)
        if area >= 50:
            new_contours.append(contour)

    new_contours = sorted(new_contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if len(new_contours) >= 4:
        new_contours = new_contours[:4]
        sorted_corners = []

        for i in range(4):
            x, y, w, h = cv2.boundingRect(new_contours[i])
            sorted_corners.append((x + w / 2, y + h / 2))
            sorted_corners = sorted(sorted_corners, key=lambda x: x[1])
        if sorted_corners[0][0] > sorted_corners[1][0]:
            sorted_corners[0], sorted_corners[1] = sorted_corners[1], sorted_corners[0]
        if sorted_corners[2][0] > sorted_corners[3][0]:
            sorted_corners[2], sorted_corners[3] = sorted_corners[3], sorted_corners[2]
        sorted_points = np.array(sorted_corners, np.float32)
        final_points = np.array([[0, 0], [765, 0], [0, 635], [765, 635]], np.float32)
        M = cv2.getPerspectiveTransform(sorted_points, final_points)
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
    cv2.waitKey(0)  # display window until any keypress
    cv2.destroyAllWindows()

    # Image processor loop
    while True:
        ret,frame=vid.read()
        frame=cv2.resize(frame,(765,635))
        # cv2.imshow("frame",frame)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  
        blue_lower = (90,50,70)
        blue_upper = (128,255,255) 
        # construct a mask for the color green
        maskboard=cv2.inRange(hsv,blue_lower,blue_upper)
        kernel = np.ones((5,5),np.uint8)
        maskboard = cv2.morphologyEx(maskboard.copy(), cv2.MORPH_OPEN, kernel)
        maskboard2 =cv2.dilate(maskboard,kernel,iterations = 1)
        cnts2,hierarchy = cv2.findContours(maskboard2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        detected_board = flatten_board(cnts2,frame)
        counter += 1

        if counter>20:
            cv2.waitKey(0)  # display window until any keypress
            cv2.destroyAllWindows()
            return detected_board
