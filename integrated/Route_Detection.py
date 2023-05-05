"""
Code written by: Norman Cheen
Automated Route Detection System
ts20365@bristol.ac.uk
University of Bristol
"""

import sys
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize_3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

from Board_Detection import board_detection
from Serial import serial_read

def find_target_contour_area(bin_image, template):
    """
    This function uses template matching to detect and locate the arrow on the maze and
    find the pixel coordinates that lie within route of the maze.
    :param bin_image : a binary (black and white) image of the maze
    :param template  : an image of an arrow used for template matching
    :return          : a list of pixel coordinates
    """

    # additional blurring and thresholding to smooth out edges and remove any effect of reflection
    bin_image = cv2.GaussianBlur(bin_image, (5, 5), cv2.BORDER_DEFAULT)
    ret,bin_image = cv2.threshold(bin_image, 150 , 255, cv2.THRESH_BINARY)

    # template matching to locate the position of the arrow on the maze
    res = cv2.matchTemplate(bin_image, template, cv2.TM_SQDIFF)
    # For TM_SQDIFF, Good match yields minimum value, thus min_loc is used
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 

    h, w = template.shape[::]   # height and width of the arrow
    top_left = min_loc          # coordinates of the top left corner of the arrow
    bottom_right = [top_left[0] + w, top_left[1] + h] 
    centre = [int(top_left[0] + w/2), int(top_left[1] + h/2)] # the centre of the arrow
    cv2.rectangle(img_gray_default, top_left, bottom_right, 10, 2)  # draw a black rectangle around the detected arrow

    target = np.array([centre[0]-2*w, centre[1]])   # a pixel that might be within the route
    target_range = np.array([target + x for x in range(-3,3)]) # a range of pixels that might be within the route (for buffer)
    target_range = target_range.tolist()

    return target_range

def find_route_contour(bin_image, target_range):
    """
    This function finds the contours of the route and verify the result automatically.
    It starts by finding the nth largest contour, then (n+1)th largest contour if result 
    is incorrect to finally obtain the correct contour of the route.
    :param bin_image    : a binary (black and white) image
    :param target_range : a range of pixel coordinates that might intersect the route contour
    :return             : detected contour
    """
    bin_image = bin_image.astype(np.uint8)  # Cast ndarray to unsigned 8-bit integer
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = tuple(sorted(contours, key=cv2.contourArea)) # Sort contour area in sequence
    contour_num = -1 # Change here to change the nth largest contour to start with (-1 being the largest)

    counter = 0
    while True:
        if counter < 5: 
            # Compute and display the largest contour area
            cur_contour = contours_sorted[contour_num]

            # Visualising the detected contour
            image_contour = np.copy(img_coloured)
            image_contour = cv2.drawContours(image_contour, [cur_contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

            green = [0,255,0]
            Y, X = np.where(np.all(image_contour == green,axis=2))
            cur_contour_outline = np.column_stack((X,Y))
            cur_contour_outline = cur_contour_outline.tolist()

            # # Optional: Plotting to double check the intersecting line
            # x2,y2 = target_range.T
            # plt.plot(x2,-y2,"r-")
            # plt.plot(X,-Y,"g*")
            # plt.show()
            # cv2.imshow("image_contour",image_contour)
            # cv2.waitKey(0)  # display window until any keypress
            # cv2.destroyAllWindows()
            
            if any(target in target_range for target in cur_contour_outline): # if target range intersect with current contour
                break
            else:
                contour_num -= 1
                # print("That wasn't right. I'll try again.")
                counter += 1
        else:
            print("Unable to detect route. Please adjust the lighting on the maze.")
            sys.exit(1)
    return cur_contour


def extract_route(bin_image, contour):
    """
    This function extract the route (treated as the foreground of the image),
    and return its skeleton pixel coordinates.
    :param bin_image    : a binary (black and white) image
    :param contour      : the contour of the desired object that is to be extracted (ie. the route) 
    :return             : the skeleton of the extracted object  
    """
    # create a black `mask` the same size as the original grayscale image 
    mask = np.zeros_like(bin_image)
    # fill the new mask with the shape of the largest contour in white
    cv2.fillPoly(mask, [contour], 255)

    # create a copy of the current mask
    res_mask = np.copy(mask)
    res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
    res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
    res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels

    # create a mask for obvious and probable foreground pixels
    # obvious foreground pixels will be white
    # probable foreground pixels will be black
    mask2 = np.where(
    (res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD),
    255,
    0
    ).astype('uint8')

    # create `new_mask3d` from `mask2` but with 3 dimensions instead of 2
    new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
    mask3d = new_mask3d
    mask3d[new_mask3d > 0] = 255.0
    mask3d[mask3d > 255] = 255.0

    # apply Gaussian blurring to smoothen out the edges again
    mask3d = cv2.GaussianBlur(mask3d, (5, 5), 0)

    # skeletonise the image
    skeleton = skeletonize_3d(mask3d)

    return skeleton

def find_route_coordinates(skeleton_image):
    """
    This function finds the pixel coordinates of the route
    :param skeletonised_image   : a skeletonised image of the route 
    :return                     : pixel coordinates of the route
    """
    # Obtain the coordinates of the pixels forming the route (in green)
    green = [0,255,0]
    Y, X = np.where(np.all(skeleton_image==green,axis=2))
    coordinates = np.column_stack((X,Y))

    return coordinates

def find_end_points(skeleton_image):
    """
    This function finds the start and end points of the route using
    Hit-or-miss Transform
    :param image: skeleton image of the route
    :return     : coordinates of start and end points of the route
    """
    # Finding the start and end points of the route
    # kernels to find all possible endpoint patterns:
    k1 = np.array(([0, 0, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    k2 = np.array(([-1, -1, -1], [0, 1, -1], [0, -1, -1]), dtype="int")
    k3 = np.array(([-1, -1, 0],  [-1, 1, 0], [-1, -1, -1]), dtype="int")
    k4 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, 0, 0]), dtype="int")

    # convert BGR image to gray scale
    route_skeleton_gray = cv2.cvtColor((skeleton_image).astype('uint8'),cv2.COLOR_BGR2GRAY)

    # perform hit-miss transform for every kernel (output type = array)
    o1 = cv2.morphologyEx(route_skeleton_gray, cv2.MORPH_HITMISS, k1)
    o2 = cv2.morphologyEx(route_skeleton_gray, cv2.MORPH_HITMISS, k2)
    o3 = cv2.morphologyEx(route_skeleton_gray, cv2.MORPH_HITMISS, k3)
    o4 = cv2.morphologyEx(route_skeleton_gray, cv2.MORPH_HITMISS, k4)

    # add the results of all the 4 outputs
    hitmiss_out = o1 + o2 + o3 + o4

    # find none-zero points (ie. end points) First element = start, Second element = end
    end_points = np.argwhere(hitmiss_out)

    # # Optional: Ploting the determined end-points for checking
    # img_end_points = route_skeleton.copy()
    # for pt in end_points:
    #     pt[1],pt[0] = pt[0],pt[1] # invert so that format == [x,y]
    #     img_end_points = cv2.circle(img_end_points, (pt[0], pt[1]), 15, (0,0,255), -1)
    #     # print(pt)
    # cv2.imshow('endpoint',img_end_points)
    # print("\n The start and end points are:\n" , end_points)

    return end_points

def find_neighbour(cur_coordinates):
    """
    This function finds the 8 neighbours surrounding a point and return their coordinates in an array
    :param cur_coordinates  : coordinates of a point 
    :return                 : an array consisting of the coodinates of its neighbours
    """
    neighbours = []
    for n in range(-1,2):
        for m in range(-1,2):
            pt = [n,m]
            neighbours.append(pt)
    neighbours.remove([0,0])
    neighbours = np.array(neighbours)
    neighbours +=  cur_coordinates
    return neighbours

def sort_route(start_pt,end_pt,all_pts):
    """
    This function sorts the coordinates of points between the start and the end points in order
    and return them in an array. 
    :param start_pt: the coordinates of the starting point [array]
    :param end_pt  : the coordinates of the starting point
    :param all_pts : the coordinates of all the points in the route
    :return        : a list of sorted coordinates of the route
    """
    cur_pt = start_pt.tolist()
    coordinates = (all_pts.tolist())

    sorted_coordinates = [cur_pt]
    while np.array_equal(cur_pt, end_pt) == False:
        cur_neighbour = find_neighbour(cur_pt).tolist()             # get neighbours and convert to a list
        next_pt = [x for x in cur_neighbour if x in coordinates][0] # common element of 2 lists
        # print(next_pt)
        sorted_coordinates.append(next_pt)      
        coordinates.remove(next_pt)         # remove determined point
        cur_pt = next_pt

    return sorted_coordinates


def route_detection_main():
    """
    This is the main function that initiates the route detection process.
    """
    global img_gray_default, img_gray, img_coloured, img_bin, target_route, route_contour, route_skeleton, route_coordinates, end_points, sorted_route_coordinates

    # Read Image-----------------------------------------------------------------------------------------------------------------------------------------------
    detected_board = board_detection()    # raw, cropped image of the maze
    img_gray_default = cv2.cvtColor(detected_board,cv2.COLOR_BGR2GRAY)


    tem_path = r'C:\Desktop\Desktop Summary\EEE\groupproject\arrow.png'  # path of the template image
    tem = cv2.imread(tem_path, 0)

    # # Resize image
    dim = (765, 635) # dimension of image (scaling factor of 2.5 to 255)
    img_gray_default = cv2.resize(img_gray_default, dim, interpolation = cv2.INTER_AREA)

    # Pre-processing of image
    # blur image to smooth the edges and reduce noise caused by reflection (odd value only)
    img_gray = cv2.GaussianBlur(img_gray_default, (5, 5), cv2.BORDER_DEFAULT) 
    img_coloured = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB) # coloured image
    img_bin =cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2) # binary (black and white image)



    target_route = find_target_contour_area(img_bin, tem)

    route_contour = find_route_contour(img_bin, target_route) # Find the contour of the ball's route

    route_skeleton = extract_route(img_bin, route_contour)

    route_coordinates = find_route_coordinates(route_skeleton)

    end_points = find_end_points(route_skeleton)

    sorted_route_coordinates = sort_route(end_points[0], end_points[1], route_coordinates)

    sorted_route_coordinates_filtered = [sorted_route_coordinates[i] for i in range(len(sorted_route_coordinates)) if i%46 == 0]

    # wrting the route coordinates into a file
    file = open("coordinates.txt","wb")
    pickle.dump(sorted_route_coordinates_filtered, file)

    # setting up a route packet for serial communication
    payload_length = len(sorted_route_coordinates_filtered)
    route_packet = [2,5,(payload_length & 0xff),(payload_length >> 8)&0xff]
    for i in range(payload_length):
        for j in range(2):
            route_packet.append(int(sorted_route_coordinates_filtered[i][j]/3))
    route_packet.append(3)
    route_packet = bytearray(route_packet)
    serial_read(route_packet)

    return
