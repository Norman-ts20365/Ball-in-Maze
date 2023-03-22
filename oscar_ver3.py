    count=0
    centers=[]
    arrow=0
    arrowcount=0
    a=0
    b=0
    check=0

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 75 and count<100:
            count=0
            x,y,w,h =cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            center=(int(x+w/2),int(y+h/2))
            xcor=int(x+w/2)
            ycor=int(y+h/2)
            print(center)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            centers.append(center)
            arrowcount=arrowcount+1
            min=datetime.now().strftime("%M")
            min=int(min)
            sec=datetime.now().strftime("%S")
            sec=int(sec)
            microsec=datetime.now().strftime("%f")
            microsec1=int(str(microsec)[:2])
            microsec2=int(str(microsec)[2:4])
            microsec3=int(str(microsec)[4:])
            global test_packet
            test_packet=([2,4,xcor,ycor,min,sec,microsec1,microsec2,microsec3,3])
            return test_packet

        
        elif cv2.contourArea(c) < 75 :
            center="ball disappeared"
            test_packet=[]      
            return test_packet
