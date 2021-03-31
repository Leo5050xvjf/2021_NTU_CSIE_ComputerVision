import cv2
import numpy as np



def binary():
    img = cv2.imread("./lena.bmp")
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # cv2.imshow("Binary",thresh1)
    # cv2.waitKey(0)
    cv2.imwrite("HW1_1_03.jpg",thresh1)

if __name__  == "__main__":
    binary()

