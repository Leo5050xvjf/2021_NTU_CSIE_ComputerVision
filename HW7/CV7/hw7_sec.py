import cv2
import  numpy as np

def sampling_and_binarize_():
    img = cv2.imread("./lena.bmp",0)
    h,w = img.shape
    template = np.zeros((64,64))
    for height in range(int(h/8)-1):
        for weight in range(int(w//8)-1):
            template[height,weight] = img[height*8,weight*8]
    template = np.uint8(template)
    h1,w1 = template.shape

    img =np.reshape(template,(1,h1*w1))
    img[img >127] = 255
    img[img <=127] = 0
    img =np.reshape(img,(h1,w1))
    img = np.uint8(img)
    return img
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
        # print("num",num)
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
    template = np.zeros((64,64),dtype = int)
    img =np.pad(img,(1,1))
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if img[height,weight] != 0:
                num = count_number(img[height-1:height+2,weight-1:weight+2])
                template[height-1,weight-1] = num

    # np.set_printoptions(linewidth=200,threshold = 100000)
    # template = np.reshape(template,(64,64))
    # print(template)
    template = np.uint8(template)
    # template 內是 yokoi number 0~5 組成的影像
    return template
def thinning(img):

    h,w = img.shape
    img =np.pad(img,(1,1))
    template = np.zeros((64,64),dtype=str)
    sum_ = 0
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if img[height,weight] !=0:
                if img[height + 1, weight] == 1:
                    sum_ += 1
                if img[height - 1, weight] == 1:
                    sum_ += 1
                if img[height, weight + 1] == 1:
                    sum_ += 1
                if img[height, weight - 1] == 1:
                    sum_ += 1
                if sum_<1 or img[height,weight] != 1:
                    template[height-1,weight-1] = 'q'
                if sum_>=1 and img[height,weight] == 1:
                    template[height-1,weight-1] = 'p'

    img = img[1:h+1,1:w+1]
    template = np.pad(template,(1,1))
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if template[height,weight] == 'p':
                # 目前座標是＋1狀態
                slice_img = template[height-1:height+2,weight-1:weight+2]
                q= del_or_not(slice_img)
                if q == 1:
                    template[height,weight] = '0'
                    img[height-1,weight-1] = 0
    for height in range(h):
        for weight in range(w):
            if img[height,weight]!=0:
                img[height, weight] = 255
    img =np.uint8(img)
    return img
def del_or_not(img):
    mask1 = np.array([['p','p','p','p']])
    q = 0
    if img[1,2] =='p':

        slice1 = img[0:2,1:3]
        slice1 = np.reshape(slice1,(1,4))
        print("slice", slice1)
        print("mask",mask1)

        if not (slice1 == mask1).all():
            q += 1
    # 上
    if img[0,1] =='p':
        slice2 = img[0:2,0:2]
        slice2 = np.reshape(slice2,(1,4))

        if not (slice2 == mask1).all():
            q += 1
    # 左
    if img[1,0] =='p':
        slice3 = img[1:3,0:2]
        slice3 = np.reshape(slice3,(1,4))

        if not (slice3 == mask1).all():
            q += 1
    # 下
    if img[2,1] =='p':
        slice4 = img[1:3,1:3]
        slice4 = np.reshape(slice4,(1,4))

        if not (slice4 == mask1).all():
            q += 1
    return q
img = sampling_and_binarize_()
for i in range(100):

    img = yokoi(img)
    img = thinning(img)
    cv2.namedWindow("enhanced", 0);
    cv2.resizeWindow("enhanced", 640, 480);
    cv2.imshow("enhanced", img)
    cv2.waitKey(0)
img = yokoi(img)
img = thinning(img)
cv2.namedWindow("enhanced", 0);
cv2.resizeWindow("enhanced", 640, 480);
cv2.imshow("enhanced", img)
cv2.waitKey(0)
img = yokoi(img)
img = thinning(img)
cv2.namedWindow("enhanced", 0);
cv2.resizeWindow("enhanced", 640, 480);
cv2.imshow("enhanced", img)
cv2.waitKey(0)
img = yokoi(img)
img = thinning(img)
cv2.namedWindow("enhanced", 0);
cv2.resizeWindow("enhanced", 640, 480);
cv2.imshow("enhanced", img)
cv2.waitKey(0)









