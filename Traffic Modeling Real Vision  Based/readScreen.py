import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'top': 140, 'left': 100, 'width': 884, 'height': 900}

sct = mss()

fgbg = cv2.createBackgroundSubtractorMOG2()

checker  = np.zeros((80),dtype=int)



def warped_simulation(rect,frame):
    dst = np.array([(125, 0), (375, 0), (500, 800), (0, 800)], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (500, 800))
    #cv2.imshow('warped', warped)
    return warped


def tail_length(lane):

    fgmask = fgbg.apply(lane)
    #cv2.imshow("bog",fgmask)
    start = 800

    for i in range(80):
        density = lane[ start-10:start ,62:438]
        cv2.imshow("density",density)
        d_gray = cv2.cvtColor(density, cv2.COLOR_BGR2GRAY)
        white = cv2.countNonZero(d_gray)
        print(" ", white)
        #start +=10


        if white > 1500:
            checker[i] = 1
        else:
            checker[i] = 0
        start -= 10


    tail = 800

    for i in range(77):
        over = 1
        for j in range(i, i + 3):
            if checker[j] == 1:
                over = 0
                break

        if over == 1:
            tail = i
            break

    print(checker)
    print(tail)

    return tail


while 1:
    sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    image = np.array(img)

    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # rotate the image by 180 degrees
    M180 = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated180 = cv2.warpAffine(image, M180, (w, h))

    M90 = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated90 = cv2.warpAffine(image, M90, (w, h))

    M270 = cv2.getRotationMatrix2D(center, 270, 1.0)
    rotated270 = cv2.warpAffine(image, M270, (w, h))

    upper_lane =  np.array([(428,39),(517,39),(517,351),(428,351)], dtype = "float32")

    # coordinates on 180 rotation image
    lower_lane =  np.array([(456,31),(544,31),(544,349),(456,349)], dtype = "float32")

    #coordinates on 90 rotation image
    right_lane =  np.array([(445,36),(533,36),(533,363),(445,363)], dtype = "float32")

    #cordinates on 270 rotation image
    left_lane  =  np.array([(439,34),(528,34),(528,334),(439,334)], dtype = "float32")



    warp_upperlane = warped_simulation(upper_lane,image)
    cv2.imshow("upper",warp_upperlane)

    warp_lowerlane = warped_simulation(lower_lane, rotated180)
    #cv2.imshow("lower", warp_lowerlane)

    warp_rightlane = warped_simulation(right_lane, rotated90)
    #cv2.imshow("right", warp_rightlane)

    warp_leftlane = warped_simulation(left_lane, rotated270)
    #cv2.imshow("left", warp_leftlane)

    tail_length_upper = tail_length(warp_upperlane)
    tail_length_lower = tail_length(warp_lowerlane)
    tail_length_right = tail_length(warp_rightlane)
    tail_length_left  = tail_length(warp_leftlane)

    print("Tail lengths - ")

    print("Upper Tail ",tail_length_upper)
    print("lower Tail ", tail_length_lower)
    print("right Tail ", tail_length_right)
    print("left Tail ", tail_length_left)

    print("  end")




    #cv2.imshow('test', image)
    #cv2.imshow("180",rotated180)
    #cv2.imshow("90",rotated90)
    #cv2.imshow("270",rotated270)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break





'''
def tail_length(frame):

    return tail_length
'''