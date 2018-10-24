import numpy as np
import cv2
from mss import mss
from PIL import Image

#mon = {'top': 140, 'left': 50, 'width':1800, 'height': 900}

mon_right = {'top': 119, 'left':933, 'width': 884, 'height': 900}
mon_left = {'top': 140, 'left': 100, 'width': 884, 'height': 900}

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


def leftsubtractImage(image, SIDE):
    if SIDE == 'upper':
        bg = cv2.imread('cross_left_upper.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("sub_upper", fgmask)

    elif SIDE == 'lower':
        bg = cv2.imread('cross_left_lower.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_lower", fgmask)

    elif SIDE == 'right':
        bg = cv2.imread('cross_left_right.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_right", fgmask)

    elif SIDE == 'left':
        bg = cv2.imread('cross_left_left.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_left", fgmask)

    return fgmask




def rightsubtractImage(image, SIDE):
    if SIDE == 'upper':
        bg = cv2.imread('cross_right_upper.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("sub_upper", fgmask)

    elif SIDE == 'lower':
        bg = cv2.imread('cross_right_lower.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_lower", fgmask)

    elif SIDE == 'right':
        bg = cv2.imread('cross_right_right.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_right", fgmask)

    elif SIDE == 'left':
        bg = cv2.imread('cross_right_left.png')
        fgmask = image - bg
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bog_left", fgmask)

    return fgmask




def tail_length(mask):
    checker = np.zeros((80), dtype=int)
    start = 800

    for i in range(80):
        density = mask[start - 10:start, 0:500]

        white = cv2.countNonZero(density)
        #print(" ", white)
        # start +=10

        if white > 1000:
            checker[i] = 1
        else:
            checker[i] = 0
        start -= 10

    tail = 80

    for i in range(65):
        over = 1
        for j in range(i, i + 15):
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


def LeftgetScreenImage():
    sct.get_pixels(mon_left)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    image = np.array(img)
    #cv2.imshow("left",image)

    return image

def RightgetScreenImage():
    sct.get_pixels(mon_right)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    image = np.array(img)
    #cv2.imshow("right",image)

    return image


def getCenter(image):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    return h, w, center

#done callibration

#left intersection



def leftgetUpperQlength():
    image = LeftgetScreenImage()
    #cv2.imshow("test",image)
    upper_lane = np.array([(446, 19), (535, 19), (535, 331), (446, 331)], dtype="float32")
    warp_upperlane = warped_simulation(upper_lane, image)
    #cv2.imshow("testupper",warp_upperlane)
    mask_upper = leftsubtractImage(warp_upperlane, "upper")
    tail_length_upper = tail_length(mask_upper)

    return tail_length_upper


def leftgetLowerQlength():#
    image = LeftgetScreenImage()
    h, w, center = getCenter(image)
    M180 = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated180 = cv2.warpAffine(image, M180, (w, h))
    lower_lane = np.array([(440, 33), (528, 33), (528, 351), (440, 351)], dtype="float32")
    warp_lowerlane = warped_simulation(lower_lane, rotated180)
    #cv2.imshow("llowertest",warp_lowerlane)
    mask_lower = leftsubtractImage(warp_lowerlane, "lower")
    tail_length_lower = tail_length(mask_lower)

    return tail_length_lower


def leftgetRightQlength():
    image = LeftgetScreenImage()
    h, w, center = getCenter(image)
    M90 = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated90 = cv2.warpAffine(image, M90, (w, h))
    right_lane = np.array([(429, 30), (523, 30), (523, 343), (429, 343)], dtype="float32")
    warp_rightlane = warped_simulation(right_lane, rotated90)
    #cv2.imshow("lrtest",warp_rightlane)
    mask_right = leftsubtractImage(warp_rightlane, "right")
    tail_length_right = tail_length(mask_right)

    return tail_length_right


def leftgetLeftQlength():
    image = LeftgetScreenImage()
    h, w, center = getCenter(image)
    M270 = cv2.getRotationMatrix2D(center, 270, 1.0)
    rotated270 = cv2.warpAffine(image, M270, (w, h))
    left_lane = np.array([(454, 42), (543, 42), (543, 342), (454, 342)], dtype="float32")
    warp_leftlane = warped_simulation(left_lane, rotated270)
    #cv2.imshow("lltest",warp_leftlane)
    mask_left = leftsubtractImage(warp_leftlane, "left")
    tail_length_left = tail_length(mask_left)
    return tail_length_left




#right intersection here

#done
def rightgetUpperQlength():
    image = RightgetScreenImage()
    #cv2.imshow("test",image)
    upper_lane = np.array([(500, 31), (599, 31), (599, 351), (500, 351)], dtype="float32")
    #cv2.imshow("upper",upper_lane)
    warp_upperlane = warped_simulation(upper_lane, image)
    #cv2.imshow("upper",warp_upperlane)
    mask_upper = rightsubtractImage(warp_upperlane, "upper")
    #cv2.imshow("upper", mask_upper)
    tail_length_upper = tail_length(mask_upper)

    return tail_length_upper
#done

def rightgetLowerQlength():
    image = RightgetScreenImage()
    h, w, center = getCenter(image)
    M180 = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated180 = cv2.warpAffine(image, M180, (w, h))
    lower_lane = np.array([(378, 31), (466, 31), (466, 340), (378, 340)], dtype="float32")
    #cv2.imshow("lower", lower_lane)
    warp_lowerlane = warped_simulation(lower_lane, rotated180)
    #cv2.imshow("lower", warp_lowerlane)
    mask_lower = rightsubtractImage(warp_lowerlane, "lower")
    #cv2.imshow("lower", mask_lower)
    tail_length_lower = tail_length(mask_lower)

    return tail_length_lower

#done
def rightgetRightQlength():
    image = RightgetScreenImage()
    h, w, center = getCenter(image)
    M90 = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated90 = cv2.warpAffine(image, M90, (w, h))
    right_lane = np.array([(450, 0), (540, 0), (540, 281), (450, 281)], dtype="float32")
    #cv2.imshow("right", right_lane)
    warp_rightlane = warped_simulation(right_lane, rotated90)
    #cv2.imshow("right", warp_rightlane)
    mask_right = rightsubtractImage(warp_rightlane, "right")
    #cv2.imshow("right", mask_right)
    tail_length_right = tail_length(mask_right)

    return tail_length_right


def rightgetLeftQlength():
    image = RightgetScreenImage()
    h, w, center = getCenter(image)
    M270 = cv2.getRotationMatrix2D(center, 270, 1.0)
    rotated270 = cv2.warpAffine(image, M270, (w, h))
    left_lane = np.array([(431, 38), (520, 38), (520, 412), (430, 412)], dtype="float32")
    #cv2.imshow("left", left_lane)
    warp_leftlane = warped_simulation(left_lane, rotated270)
    cv2.imshow("left", warp_leftlane)
    mask_left = rightsubtractImage(warp_leftlane, "left")
    #cv2.imshow("left", mask_left)
    tail_length_left = tail_length(mask_left)
    return tail_length_left


if __name__ == '__main__':
    while 1:

        #image = getScreenImage()
        #cv2.imshow("test",image)

        #left intersection

        tail_length_upper = leftgetUpperQlength()
        tail_length_lower = leftgetLowerQlength()
        tail_length_right = leftgetRightQlength()
        tail_length_left = leftgetLeftQlength()



        #
        print("Tail lengths Left - ")

        print("Upper Tail ", tail_length_upper)
        print("lower Tail ", tail_length_lower)
        print("right Tail ", tail_length_right)
        print("left Tail ", tail_length_left)


        #right intersecrtion

        tail_length_upper = rightgetUpperQlength()
        tail_length_lower = rightgetLowerQlength()
        tail_length_right = rightgetRightQlength()
        tail_length_left = rightgetLeftQlength()

        print("Tail lengths right - ")

        print("Upper Tail ", tail_length_upper)
        print("lower Tail ", tail_length_lower)
        print("right Tail ", tail_length_right)
        print("left Tail ", tail_length_left)

        #
        print("  end")

        # cv2.imshow('test', image)
        # cv2.imshow("180",rotated180)
        # cv2.imshow("90",rotated90)
        # cv2.imshow("270",rotated270)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    '''
    def tail_length(frame):

        return tail_length
    '''
