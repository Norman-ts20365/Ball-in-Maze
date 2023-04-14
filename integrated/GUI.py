import tkinter as tk, threading
from PIL import Image, ImageTk
import cv2
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(0) # Ensure screen size is accurate
from subprocess import call
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import serial
import imutils

from Route_Detection import route_detection_main
from Maze_Adjustment import user_command
from trial import trial



bg_colour = "#000000"
ag_colour = "#122222"

def start():
    global is_running
    global start_time
    if not is_running:
        is_running = True
        start_time = time.time()
        # update_time()
 
# Stop the stopwatch
def stop():
    global is_running
    is_running = False
 
# Update the elapsed time
def update_time():
    if is_running:
        elapsed_time = (time.time() - start_time)
        elapsed_time = str(timedelta(seconds=elapsed_time))
        elapsed_time = elapsed_time[2:10]
        time_label.config(text=elapsed_time)
        time_label.after(50, update_time)


def animation_plot(x_cor, y_cor):
        # Create a figure and axis
        # fig, ax = plt.subplots()
        fig = plt.Figure()
        canvas = FigureCanvasTkAgg(fig, master=frame2)
        canvas.get_tk_widget().grid(column=0,row=1,sticky=tk.N)

        ax = fig.add_subplot(111)

        fig.patch.set_facecolor('black')

        # Set the axis limits
        ax.set_xlim(0, 765)
        ax.set_ylim(0, 635)
        ax.set_ylim(ax.get_ylim()[::-1])      # invert the axis
        ax.set_axis_off()
        ax.set_facecolor('black')

        # Create a line object
        line, = ax.plot([], [], lw=4)

        # Define the initialization function
        def plt_init():
            line.set_data([], [])
            return line,

        # Define the update function
        def update(frame):
            # Get the x and y values up to the current frame number
            x_data = x_cor[:frame+1]
            y_data = y_cor[:frame+1]
            
            # Set the data for the line object
            line.set_data(x_data, y_data)
            return line,

        # Create the animation object
        ani = animation.FuncAnimation(fig, update, interval=0, frames=len(x_cor), init_func=plt_init, blit=True, repeat=False)
        frame2.mainloop()
        # Display the animation
        # plt.show()

coordinates = route_detection_main()

def load_frame1():
    frame1.grid_propagate(0)  # so that parent frame stays the same size and not following the things inside
    frame1.tkraise()

    # #Insert as background image
    # img = cv2.imread(r'C:\Users\Asus\Desktop\CodeGP3\path_detection\bnw.png')
    # im = Image.fromarray(img)
    # im=im.resize((ws, hs))
    # imgtk = ImageTk.PhotoImage(image=im)
    # logo_widget = tk.Label(frame1, image=imgtk, bg=bg_colour)
    # logo_widget.image = imgtk
    # logo_widget.pack()

    frame1.columnconfigure(0,weight=1)  # weight = 0 means remain, 1 onwards means expand following the frame
    # frame1.rowconfigure(0,weight=1)
    # frame1.rowconfigure(1,weight=1)
    # frame1.rowconfigure(2,weight=1)
    frame1.columnconfigure(1,weight=1)  # weight = 0 means remain, 1 onwards means expand following the frame
    frame1.columnconfigure(2,weight=1)  # weight = 0 means remain, 1 onwards means expand following the frame
    frame1.columnconfigure(3,weight=1)  # weight = 0 means remain, 1 onwards means expand following the frame

    tk.Label(
            frame1, 
            text="BALL IN MAZE SOLVER",
            bg=bg_colour,
            fg="white",
            font=("TkMenuFont", 45)
            ).grid(row=0, column=0, rowspan=1, columnspan=4, pady=165)

    tk.Button(
            frame1,
            text="X+",
            font=("TkHeadingFont", 30),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('d')] #,call(["python", "longest_line_detection.py"])]    # Determine what the button would do (just paste the function in)
            ).grid(row=1, column=0,sticky=tk.N) # should be able to adjust according to screen size
    
    tk.Button(
            frame1,
            text="X-",
            font=("TkHeadingFont", 30),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('a')] #,call(["python", "longest_line_detection.py"])]    # Determine what the button would do (just paste the function in)
            ).grid(row=1, column=1,sticky=tk.N) # should be able to adjust according to screen size
    
    tk.Button(
            frame1,
            text="Y+",
            font=("TkHeadingFont", 30),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('w')] #,call(["python", "longest_line_detection.py"])]    # Determine what the button would do (just paste the function in)
            ).grid(row=1, column=2,sticky=tk.N) # should be able to adjust according to screen size
    
    tk.Button(
            frame1,
            text="Y-",
            font=("TkHeadingFont", 30),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('s')] #,call(["python", "longest_line_detection.py"])]    # Determine what the button would do (just paste the function in)
            ).grid(row=1, column=3,sticky=tk.N) # should be able to adjust according to screen size

    tk.Button(
            frame1,
            text="NEXT",
            font=("TkHeadingFont", 30),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[load_frame2()] #,call(["python", "longest_line_detection.py"])]    # Determine what the button would do (just paste the function in)
            ).grid(row=2, column=0,rowspan=1, columnspan=4, pady=100, sticky=tk.S) # should be able to adjust according to screen size

    # tk.Label(
    #         frame1, 
    #         # text="Place any maze onto the solver, place the ball at the starting point and let us do the rest.",
    #         text=" ",
    #         bg=bg_colour,
    #         fg="white",
    #         font=("TkMenuFont", 14)
    #         ).grid(row=2, column=0, sticky=tk.N)

def load_frame2():
    # clear_widgets(frame1)

    # stack frame 2 above frame 1
    frame1.destroy()
    frame2.grid_propagate(0)
    frame2.tkraise()
    x_cor = [x[0] for x in coordinates]
    y_cor = [y[1] for y in coordinates]

#     img = cv2.imread(r'C:\Users\Asus\Desktop\CodeGP3\path_detection\bgimg.png')
#     im = Image.fromarray(img)
#     imgtk = ImageTk.PhotoImage(image=im)
#     logo_widget = tk.Label(frame2, image=imgtk, bg=bg_colour)
#     logo_widget.image = imgtk
#     logo_widget.pack()

    frame2.columnconfigure(0,weight=1)  # weight = 0 means remain, 1 onwards means expand following the frame
#     frame2.rowconfigure(0,weight=1)
    frame2.rowconfigure(1,weight=1)
    frame2.rowconfigure(2,weight=1)

    frame2.after(10000,lambda: [start(),load_frame3()]) # Transition to next frame after x milliseconds, function will run in order

    tk.Label(
            frame2, 
            text="Route Detected",
            bg=bg_colour,
            fg="white",
            font=("TkHeadingFont", 40)
            ).grid(row=0, column=0, pady=50)  # should be able to adjust according to screen size
    
    animation_plot(x_cor, y_cor)
    
    # img = cv2.imread(r'C:\Users\Asus\Desktop\CodeGP3\path_detection\maze_image_hard.png')
    # im = Image.fromarray(img)
    # im=im.resize((950, 700))
    # imgtk = ImageTk.PhotoImage(image=im)
    # logo_widget = tk.Label(frame2, image=imgtk, bg=bg_colour)
    # logo_widget.image = imgtk
    # logo_widget.grid(row=2,column=0)

    # frame2.after(5000,lambda: [start(),load_frame3()]) # Transition to next frame after x milliseconds, function will run in order


def load_frame3():
    # clear_widgets(frame1)

    # stack frame 2 above frame 1
    frame2.destroy()
    frame3.grid_propagate(0)
    frame3.tkraise()
    
    global time_label
    time_label = tk.Label(frame3, text="00:00:00",bg=bg_colour,fg="white", font=("Helvetica", 48))
    time_label.grid(row=1, column=2, sticky=tk.N)
    update_time()

    tk.Label(
            frame3, 
            text="Solving Maze",
            bg=bg_colour,
            fg="white",
            font=("TkMenuFont", 40)
            ).grid(row=0, column=1, pady=20, sticky=tk.NSEW)

    tk.Button(
            frame3,
            text="STOP",
            font=("TkHeadingFont", 20),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[sys.exit(1)]    # Determine what the button would do (just paste the function in)
            ).grid(row=2, column=2) # should be able to adjust according to screen size
    
    tk.Button(
            frame3,
            text="RESET",
            font=("TkHeadingFont", 20),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
        #     command=lambda:[load_frame2()] #,call(["python", "longest_line_detection.py"])]    # Determine what the button would do (just paste the function in)
            ).grid(row=3, column=2) # should be able to adjust according to screen size

    # img = cv2.imread(r'C:\Users\Asus\Desktop\CodeGP3\path_detection\maze_image_hard.png')
    # im = Image.fromarray(img)
    # im=im.resize((950, 700)) 
    # imgtk = ImageTk.PhotoImage(image=im)
    # logo_widget = tk.Label(frame3, image=imgtk, bg=bg_colour)
    # logo_widget.image = imgtk
    # logo_widget.grid(row=1,column=0,rowspan=3, columnspan=2, padx=100)


#     vid = cv2.VideoCapture(r"C:\Users\Asus\Documents\Zoom\2022-11-04 23.45.05 Norman's Zoom Meeting\norman.mp4")
    
    label_widget = tk.Label(frame3)
    label_widget.grid(row=1,column=0,rowspan=3, columnspan=2, padx=100)
    def open_camera():
        # Capture the video frame by frame
        # _, frame = vid.read()
        frame = trial()
        frame = imutils.resize(frame, width=1000)
    
        # Convert image from one color space to other
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
        # Capture the latest frame and transform to image
        captured_image = Image.fromarray(opencv_image)

        # captured_image=captured_image.resize((950, 700))
    
        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)
    
        # Displaying photoimage in the label
        label_widget.photo_image = photo_image
    
        # Configure image in the label
        label_widget.configure(image=photo_image)
    
        # Repeat the same process after every 10 seconds
        label_widget.after(10, open_camera)

    # open_camera()
    thread = threading.Thread(target=open_camera)
    # thread = threading.Thread(target=stream(my_label))
    thread.daemon = 1
    thread.start()
    frame3.mainloop()




# initiallise app -------------------------------------------------------------------------
root = tk.Tk()
root.title("Ball In Maze Solver")
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen
root.geometry('{}x{}'.format(ws, hs))
# root.columnconfigure(0,weight=1)
# root.rowconfigure(0,weight=1)

print(ws,hs)
# create a frame widgets
frame1 = tk.Frame(root, width=ws, height=hs, bg=ag_colour)
frame2 = tk.Frame(root, width=ws, height=hs, bg=bg_colour)
frame3 = tk.Frame(root, width=ws, height=hs, bg=bg_colour)

for frame in (frame1, frame2,frame3):
	# frame.grid(row=0, column=0, sticky="nesw")
        frame.pack()

load_frame1()

is_running = False

# # Display Full Screen
# root.attributes('-fullscreen', True)

# run app
root.mainloop()

#-------------------------------------------------------------------------------------
# import tkinter as tk
# import time
# from PIL import Image, ImageTk
# from datetime import timedelta

# # Start the stopwatch
# def start():
#     global is_running
#     global start_time
#     if not is_running:
#         is_running = True
#         start_time = time.time()
#         update_time()
 
# # Stop the stopwatch
# def stop():
#     global is_running
#     is_running = False
 
# # Update the elapsed time
# def update_time():
#     if is_running:
#         elapsed_time = (time.time() - start_time)
#         elapsed_time = str(timedelta(seconds=elapsed_time))
#         elapsed_time = elapsed_time[2:10]
#         time_label.config(text=elapsed_time)
#         # time_label.config(text="{:.2f}".format(elapsed_time))
#         time_label.after(50, update_time)
 
# # Create the main window
# window = tk.Tk()
# # # Specify screen size of the Application window
# window.geometry('350x400')
# # # Specify application name
# window.title("Stopwatch")
# # window.config(bg='#299617')
 
# # Read image using PIL library
# # im = Image.open('stopwatch.png')
# # # Convert image to tkinter format
# # bg = ImageTk.PhotoImage(im)
# # # Display image as application background
# # img = tk.Label(window, image=bg)
# # img.place(x=0, y=0)
 
# # Create the label to display the elapsed time
# time_label = tk.Label(window, text="0.0.0.00", font=("Helvetica", 48))
# time_label.place(x=110, y=190)
 
# # Create the start and stop buttons
# start_button = tk.Button(window, text="Start", width=10, command=start)
# start_button.place(x=10, y=10)
# stop_button = tk.Button(window, text="Stop", width=10, command=stop)
# stop_button.place(x=260, y=10)
 
# # Flag to track whether the stopwatch is running
# is_running = False
 
# # Run the main loop
# window.mainloop()
