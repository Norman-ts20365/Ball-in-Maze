import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d
import os
# import imutils
import serial
import sys
from time import sleep
import csv


def find_route_contour(image):
    """
    This function finds all the contours in an image and return the nth largest
    contour area (in this case, to find the path of the ball).
    :param image: a binary (black and white) image
    :return     : pixel coordinates
    """
    image = image.astype(np.uint8)  # Cast ndarray to unsigned 8-bit integer
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_sorted = tuple(sorted(contours, key=cv.contourArea)) # Sort contour area in sequence
    contour_num = -2 # Change here to change the nth largest contour to start with
    while True:
        try:
            # Compute and display the largest contour are  
            route_contour = contours_sorted[contour_num]
            image_contour = np.copy(img_coloured)
            cv.drawContours(image_contour, [route_contour], 0, (0, 255, 0), 2, cv.LINE_AA, maxLevel=1)
            print("Is this the path? Close the image window and type yes/no")
            cv.imshow('Contour', image_contour)
            cv.waitKey(0)  # display window until any keypress
            cv.destroyAllWindows()

            # Prompt user's input
            ball_route = str(input("Please enter yes/no: "))
            if ball_route == "yes":
                print("Great! Continue to run remaining code")
                break
            elif ball_route == "no":
                contour_num -= 1
                print("I'll try again!")    
                
        except ValueError:
            print("Invalid input. Please enter yes/no.")
            continue
    return route_contour


def extract_route(image, contour):
    """
    This function extract the route (treated as the foreground of the image),
    and return its skeleton pixel coordinates.
    :param image    : a binary (black and white) image
    :param contour  : the contour of the desired object that is to be extracted (ie. the route) 
    :return         : the skeleton of the extracted object  
    """
    # create a black `mask` the same size as the original grayscale image 
    mask = np.zeros_like(image)
    # fill the new mask with the shape of the largest contour
    # all the pixels inside that area will be white
    cv.fillPoly(mask, [contour], 255)

    # # Skeletonise the image
    # route_skeleton = skeletonize_3d(mask)
    # # skel = cv.cvtColor(skeleton.astype(np.uint8),cv.COLOR_GRAY2BGR)
    # cv.imshow('skeleton',route_skeleton)

    # create a copy of the current mask
    res_mask = np.copy(mask)
    res_mask[mask == 0] = cv.GC_BGD # obvious background pixels
    res_mask[mask == 255] = cv.GC_PR_BGD # probable background pixels
    res_mask[mask == 255] = cv.GC_FGD # obvious foreground pixels

    # create a mask for obvious and probable foreground pixels
    # all the obvious foreground pixels will be white and...
    # ... all the probable foreground pixels will be black
    mask2 = np.where(
    (res_mask == cv.GC_FGD) | (res_mask == cv.GC_PR_FGD),
    255,
    0
    ).astype('uint8')

    # create `new_mask3d` from `mask2` but with 3 dimensions instead of 2
    new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
    mask3d = new_mask3d
    mask3d[new_mask3d > 0] = 255.0
    mask3d[mask3d > 255] = 255.0
    # apply Gaussian blurring to smoothen out the edges a bit
    # `mask3d` is the final foreground mask (not extracted foreground image)
    mask3d = cv.GaussianBlur(mask3d, (5, 5), 0)
    # mask3d = cv.medianBlur(mask3d,5)
    cv.imshow('Foreground mask', mask3d)
    cv.waitKey(0)  # display window until any keypress
    cv.destroyAllWindows()

    # Skeletonise the image
    skeleton = skeletonize_3d(mask3d)
    cv.imshow('skeleton', skeleton)

    return(skeleton)

def find_route_coordinates(skeleton_image):
    """
    This function finds the pixel coordinates of the route
    :param skeletonised_image   : a skeletonised image with the route in green 
    :return                     : pixel coordinates of the route
    """
    # Obtain the coordinates of the pixels forming the route
    green = [0,255,0]
    Y, X = np.where(np.all(skeleton_image==green,axis=2))
    coordinates = np.column_stack((X,Y))
    print(coordinates)
    print("Total coordinates:" , len(coordinates))

    return coordinates

def find_end_points(image):
    """
    This function finds all the contours in an image and return the nth largest
    contour area (in this case, to find the path of the ball).
    :param image: skeleton image of the route
    :return     : coordinates of start and end points of the route
    """

    # Finding the starting and end points of the route
    # kernels to find all possible endpoint patterns 
    k1 = np.array(([0, 0, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    k2 = np.array(([-1, -1, -1], [0, 1, -1], [0, -1, -1]), dtype="int")
    k3 = np.array(([-1, -1, 0],  [-1, 1, 0], [-1, -1, -1]), dtype="int")
    k4 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, 0, 0]), dtype="int")

    # convert BGR image to gray scale (black and white)
    route_skeleton_gray = cv.cvtColor((route_skeleton).astype('uint8'),cv.COLOR_BGR2GRAY)

    # perform hit-miss transform for every kernel (output type = array)
    o1 = cv.morphologyEx(route_skeleton_gray, cv.MORPH_HITMISS, k1)
    o2 = cv.morphologyEx(route_skeleton_gray, cv.MORPH_HITMISS, k2)
    o3 = cv.morphologyEx(route_skeleton_gray, cv.MORPH_HITMISS, k3)
    o4 = cv.morphologyEx(route_skeleton_gray, cv.MORPH_HITMISS, k4)

    # add results of all the above 4
    hitmiss_out = o1 + o2 + o3 + o4

    # find none-zero points (ie. end points) First element = start, Second element = end
    end_points = np.argwhere(hitmiss_out)

    # ploting the determined end-points for checking
    img_end_points = route_skeleton.copy()
    for pt in end_points:
        pt[1],pt[0] = pt[0],pt[1] # invert so that format == [x,y]
        img_end_points = cv.circle(img_end_points, (pt[0], pt[1]), 15, (0,0,255), -1)
        print(pt)
    cv.imshow('endpoint',img_end_points)
    print("\n The start and end points are:\n" , end_points)

    return end_points

def find_neighbour(cur_coordinates):
    """
    This function finds the 8 neighbours surrounding a point and return their coordinates in an array
    :param cur_coordinates: the coordinates of the point that's surrounded by the neighbours
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
    """
    cur_pt = start_pt.tolist()
    coordinates = (all_pts.tolist())
    print("\n \n old, unsorted: \n \n" , coordinates)
    sorted_coordinates = [cur_pt]
    while np.array_equal(cur_pt, end_pt) == False:
        cur_neighbour = find_neighbour(cur_pt).tolist()             # get neighbours and convert to a list
        next_pt = [x for x in cur_neighbour if x in coordinates][0] # common element of 2 lists
        # print(next_pt)
        sorted_coordinates.append(next_pt)      
        coordinates.remove(next_pt)         # delete determined point
        cur_pt = next_pt
    return sorted_coordinates
    

# Read Image-----------------------------------------------------------------------------------------------------------------------------------------------
path = r'C:\Users\Asus\Desktop\CodeGP3\path_detection\maze_image.png'
img_gray = cv.imread(path,0)  # "0" means read image in gray scale
cv.imshow('Ori gray', img_gray)

img_gray = cv.GaussianBlur(img_gray, (5, 5), cv.BORDER_DEFAULT) # blur image to smooth the edges and reduce noise caused by reflection (odd value only)
cv.imshow('Ori blurred', img_gray)


# # Resize image
# scale_percent = 80 # percent of original size
# width = int(img_gray.shape[1] * scale_percent / 100)
# height = int(img_gray.shape[0] * scale_percent / 100)
# dim = (765, 725) # Alter to the suitable size for the motor controller
dim = (765, 635)
img_gray = cv.resize(img_gray, dim, interpolation = cv.INTER_AREA)

img_coloured = cv.cvtColor(img_gray,cv.COLOR_GRAY2RGB) # coloured image
img_bin =cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)
# ret,img_bin = cv.threshold(img_gray, 200 , 255, cv.THRESH_BINARY_INV) # threshold the gray image to obtain a binary (black and white image)
cv.imshow('bin', img_bin)
cv.waitKey(0)  # display window until any keypress
cv.destroyAllWindows()

#----------------------------------------------------------------------------------------------------------------------------------------------

# Find the contour of the ball's route
route_contour = find_route_contour(img_bin)

route_skeleton = extract_route(img_bin, route_contour)

route_coordinates = find_route_coordinates(route_skeleton)

end_points = find_end_points(route_skeleton)

# sort the coordinates of the route
sorted_route_coordinates = sort_route(end_points[0], end_points[1], route_coordinates)
print("\n \n sorted route is: \n \n" , sorted_route_coordinates)

plt.imshow(route_skeleton)
plt.show()
cv.waitKey(0)  # display window until any keypress
cv.destroyAllWindows()
