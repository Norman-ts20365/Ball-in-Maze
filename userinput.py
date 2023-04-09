while True:
        try:
            # Compute and display the largest contour area  
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
