import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'top': 140, 'left':1000, 'width': 884, 'height': 900}

sct = mss()

fgbg_upper = cv2.createBackgroundSubtractorMOG2(history=5000000)
fgbg_lower = cv2.createBackgroundSubtractorMOG2(history=5000000)
fgbg_right = cv2.createBackgroundSubtractorMOG2(history=5000000)
fgbg_left = cv2.createBackgroundSubtractorMOG2(history=5000000)


def warped_simulation(rect, frame):
    dst = np.array([(125, 0), (375, 0), (500, 800), (0, 800)], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (500, 800))
    # cv2.imshow('warped', warped)
    return warped


'''
def makeMask(lane, SIDE):
    if SIDE == 'upper':
        fgmask = fgbg_upper.apply(lane)
        #cv2.imshow("bog_upper", fgmask)
    elif SIDE == 'lower':
        fgmask = fgbg_lower.apply(lane)
        #cv2.imshow("bog_lower", fgmask)
    elif SIDE == 'right':
        fgmask = fgbg_right.apply(lane)
        #cv2.imshow("bog_right", fgmask)
    elif SIDE == 'left':
        fgmask = fgbg_left.apply(lane)
        #cv2.imshow("bog_left", fgmask)

    return fgmask

'''


def subtractImage(image, SIDE):
    if SIDE == 'upper':
        bg = cv2.imread('upper_screenshot.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("sub_upper", fgmask)
        # cv2.imshow("upper", image)
        # cv2.imshow("bg",bg)

    elif SIDE == 'lower':
        bg = cv2.imread('lower_screenshot.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_lower", fgmask)

    elif SIDE == 'right':
        bg = cv2.imread('right_screenshot.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_right", fgmask)

    elif SIDE == 'left':
        bg = cv2.imread('left_screenshot.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_left", fgmask)

    return fgmask


def tail_length(mask):
    checker = np.zeros((80), dtype=int)
    start = 800
    width_threshold = 200
    for i in range(80):
        density = mask[start - 10:start, 0:500]

        white = cv2.countNonZero(density)
        #print(" ", white)
        # start +=10

        if white > width_threshold:
            checker[i] = 1
        else:
            checker[i] = 0
        start -= 10

    tail = 80
    length_threshold = 30
    for i in range(80 - length_threshold):
        over = 1
        for j in range(i, i + length_threshold):
            if checker[j] == 1:
                over = 0
                break

        if over == 1:
            tail = i
            break

    #print(checker)

    #print(tail)

    if tail < 5:
        tail = 0

    return tail


def getScreenImage():
    sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    image = np.array(img)


    return image


def getCenter(image):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    return h, w, center


def getUpperQlength():
    image = getScreenImage()
    #cv2.imshow("test",image)
    upper_lane = np.array([(455, 31), (544, 31), (544, 351), (455, 351)], dtype="float32")
    #cv2.imshow("upper",upper_lane)
    warp_upperlane = warped_simulation(upper_lane, image)
    #cv2.imshow("upper",warp_upperlane)
    mask_upper = subtractImage(warp_upperlane, "upper")
    #cv2.imshow("upper", mask_upper)
    tail_length_upper = tail_length(mask_upper)

    return tail_length_upper


def getLowerQlength():
    image = getScreenImage()
    h, w, center = getCenter(image)
    M180 = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated180 = cv2.warpAffine(image, M180, (w, h))
    lower_lane = np.array([(430, 31), (520, 31), (520, 340), (430, 340)], dtype="float32")
    #cv2.imshow("lower", lower_lane)
    warp_lowerlane = warped_simulation(lower_lane, rotated180)
    #cv2.imshow("lower", warp_lowerlane)
    mask_lower = subtractImage(warp_lowerlane, "lower")
    #cv2.imshow("lower", mask_lower)
    tail_length_lower = tail_length(mask_lower)

    return tail_length_lower


def getRightQlength():
    image = getScreenImage()
    h, w, center = getCenter(image)
    M90 = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated90 = cv2.warpAffine(image, M90, (w, h))
    right_lane = np.array([(445, 36), (533, 36), (533, 330), (445, 330)], dtype="float32")
    #cv2.imshow("right", right_lane)
    warp_rightlane = warped_simulation(right_lane, rotated90)
    #cv2.imshow("right", warp_rightlane)
    mask_right = subtractImage(warp_rightlane, "right")
    #cv2.imshow("right", mask_right)
    tail_length_right = tail_length(mask_right)

    return tail_length_right


def getLeftQlength():
    image = getScreenImage()
    h, w, center = getCenter(image)
    M270 = cv2.getRotationMatrix2D(center, 270, 1.0)
    rotated270 = cv2.warpAffine(image, M270, (w, h))
    left_lane = np.array([(439, 34), (528, 34), (528, 354), (439, 354)], dtype="float32")
    #cv2.imshow("left", left_lane)
    warp_leftlane = warped_simulation(left_lane, rotated270)
    #cv2.imshow("left", warp_leftlane)
    mask_left = subtractImage(warp_leftlane, "left")
    #cv2.imshow("left", mask_left)
    tail_length_left = tail_length(mask_left)
    return tail_length_left


if __name__ == '__main__':
    while True:

        #image = getScreenImage()
        tail_length_upper = getUpperQlength()
        tail_length_lower = getLowerQlength()
        tail_length_right = getRightQlength()
        tail_length_left  = getLeftQlength()



        print("Tail lengths - ")

        print("Upper Tail ", tail_length_upper)
        print("lower Tail ", tail_length_lower)
        print("right Tail ", tail_length_right)
        print("left Tail ", tail_length_left)

        # print("  end")
        # image = getScreenImage()
        # cv2.imshow("test", image)

        #cv2.imshow('test', image)
        # cv2.imshow("180",rotated180)
        # cv2.imshow("90",rotated90)
        # cv2.imshow("270",rotated270)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    '''
    def tail_length(frame):

        return tail_length
    '''
