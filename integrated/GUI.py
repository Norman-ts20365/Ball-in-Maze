"""
Code written by: Norman Cheen
Modern Graphical User Interface
ts20365@bristol.ac.uk
"""


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
import imutils
import pickle
from time import sleep

from Maze_Adjustment import user_command
from Route_Detection import route_detection_main

# Black background colour and tiel colour declaration
bg_colour = "#000000"
tiel = "#28393a"

# Start the stopwatch
def start_timer():
    global is_running
    global start_time
    if not is_running:
        is_running = True
        start_time = time.time()
 
# Stop the stopwatch
def stop_timer():
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
    """
    This function animate the detected route.
    :param x_cor : x coordinates of the route
    :param y_cor : y coordinates of the route
    """ 
        
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
    ani = animation.FuncAnimation(fig, update, interval=10, frames=len(x_cor), init_func=plt_init, blit=True, repeat=False)
    frame2.mainloop()


def load_frame1():
    """
    This function loads and brings up frame 1 of the GUI.
    """ 

    frame1.grid_propagate(0)  # so that parent frame stays the same size and not following the things inside
    frame1.tkraise()

    frame1.columnconfigure(0,weight=1) 
    frame1.columnconfigure(1,weight=1)  
    frame1.columnconfigure(2,weight=1)  
    frame1.columnconfigure(3,weight=1)  

    # Widgets declarations:

    tk.Label(
            frame1, 
            text="BALL IN MAZE SOLVER",
            bg=bg_colour,
            fg="white",
            font=("TkMenuFont", 70)
            ).grid(row=0, column=0, rowspan=1, columnspan=4, pady=165)

    tk.Button(
            frame1,
            text="X+",
            font=("TkHeadingFont", 40),
            bg=tiel,
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('d')] #,call(["python", "longest_line_detection.py"])]    # Determine what the button would do (just paste the function in)
            ).grid(row=1, column=0,sticky=tk.N) # should be able to adjust according to screen size
    
    tk.Button(
            frame1,
            text="X-",
            font=("TkHeadingFont", 40),
            bg=tiel,
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('a')]  # Determine what the button would do
            ).grid(row=1, column=1,sticky=tk.N) 
    
    tk.Button(
            frame1,
            text="Y+",
            font=("TkHeadingFont", 40),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('w')] 
            ).grid(row=1, column=2,sticky=tk.N)
    
    tk.Button(
            frame1,
            text="Y-",
            font=("TkHeadingFont", 40),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[user_command('s')] 
            ).grid(row=1, column=3,sticky=tk.N)

    tk.Button(
            frame1,
            text="NEXT",
            font=("TkHeadingFont", 40),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[route_detection_main(), load_frame2()] 
            ).grid(row=2, column=0,rowspan=1, columnspan=4, pady=100, sticky=tk.S)


def load_frame2():
    """
    This function loads and brings up frame 2 of the GUI.
    """ 

    # stack frame 2 above frame 1
    frame1.destroy()
    frame2.grid_propagate(0)
    frame2.tkraise()
    
    # read file to obtain the route coordinates for plotting 
    file = open("coordinates.txt","rb")
    route_coordinates = pickle.load(file)
    file.close

    x_cor = [x[0] for x in route_coordinates]
    y_cor = [y[1] for y in route_coordinates]

    frame2.columnconfigure(0,weight=1)

    tk.Label(
            frame2, 
            text="Route Detected",
            bg=bg_colour,
            fg="white",
            font=("TkMenuFont", 60)
            ).grid(row=0, column=0, pady=50)  # should be able to adjust according to screen size
    
    tk.Button(
            frame2,
            text="START",
            font=("TkHeadingFont", 30),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[load_frame3()]    # Determine what the button would do (just paste the function in)
            ).grid(row=2, column=0,pady=80) # should be able to adjust according to screen size

    animation_plot(x_cor, y_cor)



def load_frame3():
    """
    This function loads and brings up frame 3 of the GUI.
    """ 

    # multiprocessing: notify a waiting process (in this case, the maze driver)
    with condition:
        condition.notify()

    # stack frame 3 above frame 2
    frame2.destroy()
    frame3.grid_propagate(0)
    frame3.tkraise()
    
    sleep(3)
    start_timer()

    global time_label
    time_label = tk.Label(frame3, text="00:00:00",bg=bg_colour,fg="white", font=("Helvetica", 48))
    time_label.grid(row=1, column=0, columnspan=2, sticky=tk.N)
    update_time()

    frame3.columnconfigure(0,weight=1)
    frame3.columnconfigure(1,weight=1)

    tk.Label(
            frame3, 
            text="Solving Maze",
            bg=bg_colour,
            fg="white",
            font=("TkMenuFont", 60)
            ).grid(row=0, column=0, columnspan=2, pady=20)

    tk.Button(
            frame3,
            text="QUIT",
            font=("TkHeadingFont", 20),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[sys.exit(1)]
            ).grid(row=2, column=0)
    
    tk.Button(
            frame3,
            text="STOP",
            font=("TkHeadingFont", 20),
            bg="#28393a",
            fg="white",
            cursor="hand2",
            activebackground="#badee2",
            activeforeground="black",
            command=lambda:[stop_timer()] 
            ).grid(row=2, column=1)


#----------------------------------------------------------------
root = tk.Tk()
root.title("Ball In Maze Solver")
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen
root.geometry('{}x{}'.format(ws, hs))

# create frame widgets
frame1 = tk.Frame(root, width=ws, height=hs, bg=bg_colour)
frame2 = tk.Frame(root, width=ws, height=hs, bg=bg_colour)
frame3 = tk.Frame(root, width=ws, height=hs, bg=bg_colour)

# For the timer:
is_running = False


def gui_main(local_condition):
     global condition
     condition = local_condition
     for frame in (frame1, frame2,frame3):
                frame.pack()
     load_frame1()

     # Display Full Screen
     root.attributes('-fullscreen', True)
     # run app
     root.mainloop()
