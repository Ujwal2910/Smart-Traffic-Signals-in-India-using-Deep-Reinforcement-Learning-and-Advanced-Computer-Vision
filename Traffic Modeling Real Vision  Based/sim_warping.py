import numpy as np
import cv2

cap = cv2.VideoCapture('traffic_laner.mkv')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        rect = np.array([(0,0),(133,0),(133,541),(0,541)], dtype = "float32")
        dst = np.array([(125,0),(250,0),(500,800),(0,800)], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (500, 800))
        cv2.imshow('frame',frame)
        cv2.imshow('warped',warped)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    else:
        break



