import numpy as np
import cv2


cap = cv2.VideoCapture('out-1.ogv')

fgbg = cv2.createBackgroundSubtractorMOG2()

checker  = np.zeros((810),dtype=int)
while(1):
    
    
        
    ret, frame = cap.read()

    roi = frame[438:465,959:1770]

    
    #print(checker)
    start = 0
    
    for i in range(810):
        density = roi[:25,start:start+1]
        d_gray = cv2.cvtColor(density, cv2.COLOR_BGR2GRAY)
        white = cv2.countNonZero(d_gray)
        print(" ",white)
        
        if white>15:
            checker[i] = 1
        else:
            checker[i] = 0
        start += 1


    print(checker)
    tail = 810
   
    for i in range(782):
        over = 1
        for j in range(i,i+28):
            if checker[j] == 1:
                over = 0
                break

        if over == 1:
            tail = i
            break




    print(tail)
            
                
                
    


    cv2.imshow("roi",roi)
    fgmask = fgbg.apply(roi)
    cv2.imshow('roi_bg',fgmask) 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()