

import numpy as np
import cv2

try:
    img = cv2.imread("./lena.bmp", cv2.IMREAD_GRAYSCALE)
except:
    print("No image file")
def dilation_(img):

    kernel = np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
    h,w = img.shape
    img = np.pad(img,(2,2))
    temp_ = np.zeros((h+4,w+4))
    # 答案放置temp_,maximum filter kernel 設置完成
    for height in range(2,h+2):
        for width in range(2,w+2):
            slice_range = img[height-2:height+3,width-2:width+3]
            max_value = np.max(slice_range * kernel)
            temp_[height,width] = max_value
    temp_ = np.uint8(temp_)
    temp_ = temp_[2:h+2,2:w+2]
    # cv2.imshow("dilation_lena",temp_)
    # cv2.waitKey(0)
    cv2.imwrite("./dilation_lena.jpg",temp_)
    return temp_


def erosion(img):

    kernel = np.array([[256,1,1,1,256],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[256,1,1,1,256]])
    h,w = img.shape
    img = np.pad(img,(2,2))
    temp_ = np.zeros((h+4,w+4))
    for height in range(2,h+2):
        for width in range(2,w+2):
            slice_range = img[height-2:height+3,width-2:width+3]
            min_value = np.min(slice_range * kernel)
            temp_[height,width] = min_value
    temp_ = np.uint8(temp_)
    temp_ = temp_[2:h+2,2:w+2]
    # cv2.imshow("dilation_lena",temp_)
    # cv2.waitKey(0)
    cv2.imwrite("./erosion_lena.jpg",temp_)
    return temp_

def opening():
    try:
        img  = cv2.imread("./lena.bmp",cv2.IMREAD_GRAYSCALE)
    except:
        print("No image file")
    e = erosion(img)
    d = dilation_(e)
    cv2.imwrite("./opening_lena.jpg",d)

    # cv2.imshow("opening",d)
    # cv2.waitKey(0)




def closing():
    try:
        img = cv2.imread("./lena.bmp", cv2.IMREAD_GRAYSCALE)
    except:
        print("No image file")
    d = dilation_(img)
    e = erosion(d)
    cv2.imwrite("./closing_lena.jpg",e)

    # cv2.imshow("opening", e)
    # cv2.waitKey(0)

dilation_()
erosion()
opening()
closing()