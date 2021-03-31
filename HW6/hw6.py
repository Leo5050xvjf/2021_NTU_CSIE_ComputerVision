

import cv2
import  numpy as np

def sampling_and_binarize_():
    img = cv2.imread("./lena.bmp",0)
    h,w = img.shape
    template = np.zeros((64,64))
    for height in range(int(h/8)-1):
        for weight in range(int(w//8)-1):
            # print(img[height*8,weight*8])
            template[height,weight] = img[height*8,weight*8]

            # template[height,weight] = img[height*8,weight*8]
    template = np.uint8(template)
    h1,w1 = template.shape
    img =np.reshape(template,(1,h1*w1))
    img[img >127] = 255
    img[img <=127] = 0
    img =np.reshape(img,(h1,w1))
    img = np.uint8(img)
    cv2.imwrite("./lena64.jpg",img)
    return img
img = sampling_and_binarize_()
def count_number(img):
    mask1 = np.array([[1,1],[0,0]])
    mask2 = np.array([[1, 0], [1, 0]])
    mask3 = np.array([[0,0], [1, 1]])
    mask4 = np.array([[0, 1], [0, 1]])
    q = 0
    r = 0
    # 右
    if img[1,2] != 0:
        slice1 = img[0:2,1:3]
        num = np.sum(np.bitwise_and(slice1,mask1))
        if num != 2:q+=1
        else:r+=1
    # 上
    if img[0,1] != 0:
        slice2 = img[0:2,0:2]
        num = np.sum(np.bitwise_and(slice2,mask2))
        if num != 2:q+=1
        else:r+=1
    # 左
    if img[1,0] != 0:
        slice3 = img[1:3,0:2]
        num  = np.sum(np.bitwise_and(slice3,mask3))
        if num != 2:q+=1
        else:r+=1
    # 下
    if img[2,1] != 0:
        slice4 = img[1:3,1:3]
        num  = np.sum(np.bitwise_and(slice4,mask4))
        if num != 2:q+=1
        else:r+=1
    if r == 4:return 5
    else:return q
def yokoi(img):
    # 回傳一個矩陣
    h,w = img.shape
    is_full = False
    template = np.zeros((64,64),dtype = int)
    img =np.pad(img,(1,1))
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if img[height,weight] != 0:
                num = count_number(img[height-1:height+2,weight-1:weight+2])
                template[height,weight] = num
    np.set_printoptions(linewidth=200,threshold = 100000)
    template = np.reshape(template,(64,64))
    print(template)
yokoi(img)



