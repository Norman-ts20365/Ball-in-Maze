from multiprocessing import Process, Queue, Manager, Value, Condition
import pickle
import serial

from GUI import gui_main, load_frame3
# from temp import testfunc
from Route_Detection import route_detection_main
from Maze_Driver import maze_driver_main


def serial_read(queue):
    # Serial object
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM4'
    ser.open()
    while True:
        if not queue.empty():
            cmd = queue.get() # sharing from another process the serial write command
            ser.write(cmd)
            print("Sent: ", cmd)
        if (ser.in_waiting > 0):
            print(ser.read_until().decode("utf-8"), end = '')
        return


if __name__ == "__main__":
    # manager = Manager()

    # multiprocessing-related functions
    queue = Queue()
    condition = Condition()

    # declare processes
    process_gui = Process(target=gui_main, args=(condition,queue,))
    # process_test = Process(target=testfunc, args=(condition,))
    process2_Maze_Driver = Process(target=maze_driver_main, args=(condition,queue,))
    # process_serial_read = Process(target=serial_read, args=(queue,))

    # run processes in parallel
    process_gui.start()
    # process_test.start()
    process2_Maze_Driver.start()
    # process_serial_read.start()


    # file = open("coordinates.txt","rb")
    # route_coordinates = pickle.load(file)
    # file.close

    # # setting up a route packet for serial communication
    # payload_length = len(route_coordinates)
    # route_packet = [2,5,(payload_length & 0xff),(payload_length >> 8)&0xff]
    # for i in range(payload_length):
    #     for j in range(2):
    #         route_packet.append(int(route_coordinates[i][j]/3))
    # route_packet.append(3)
    # route_packet = bytearray(route_packet)
    # queue.put(route_packet)



    # wait for processes to return 
    process_gui.join()
    # process_test.join()
    process2_Maze_Driver.join()
    # process_serial_read.join()
