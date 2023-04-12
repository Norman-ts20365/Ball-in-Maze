import serial
from Maze_Driver import serial_read

# Packet Pre-declaration
jog_Yplus = bytearray([2,7,119,3]) # 2,7,W,3
jog_Yminus = bytearray([2,7,115,3])
jog_Xplus = bytearray([2,7,100,3]) # STX, JOG_TYPE, D, ETX
jog_Xminus = bytearray([2,7,97,3])

def user_command(command):
    if command == 'w':
        # print("Jog y+")
        serial_read(jog_Yplus)
    elif command == 's':
        # print("Jog y-")
        serial_read(jog_Yminus)
    elif command == 'd':
        # print("Jog x+")
        serial_read(jog_Xplus)
    elif command == 'a':
        # print("Jog x-")
        serial_read(jog_Xminus)
