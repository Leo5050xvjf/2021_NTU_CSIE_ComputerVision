import cv2
import numpy as np



def upside_down():
    img = cv2.imread("./lena.bmp")
    h,w,c = img.shape
    template = np.zeros([h,w,c],np.uint8)
    for i in range(h):
        for j in range(w):
            template[511-i,j] = img[i,j]
    cv2.imwrite("HW1_1_01.jpg",template)
    return 0

# upside_down()
def right_side_left():
    img = cv2.imread("./lena.bmp")
    h, w, c = img.shape
    template = np.zeros([h, w, c], np.uint8)
    for i in range(h):
        for j in range(w):
            template[i,511-j] = img[i,j]
    cv2.imwrite("HW1_1_02.jpg",template)
    return 0
# right_side_left()

def diagonally_flip():
    img = cv2.imread("./lena.bmp")
    h, w, c = img.shape
    template = np.zeros([h, w, c], np.uint8)
    for i in range(h):
        for j in range(w):
            template[j,i] = img[i,j]
    cv2.imwrite("HW1_1_03.jpg",template)
    return 0
# diagonally_flip()

if __name__  == "__main__":
    upside_down()
    right_side_left()
    diagonally_flip()