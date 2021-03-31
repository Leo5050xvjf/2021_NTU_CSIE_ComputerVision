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
    cv2.namedWindow("enhanced", 0);
    cv2.resizeWindow("enhanced", 640, 480);
    cv2.imshow("enhanced", img)
    cv2.waitKey(0)

    return img
def BoaderOrNot(slice_img):
    boader = False
    center_ = slice_img[1,1]
    if slice_img[1,2] !=center_:
        boader =True
    if slice_img[0,1] != center_:
        boader=True
    if slice_img[1,0] != center_:
        boader = True
    if slice_img[2,1] != center_:
        boader = True
    return boader
def BI(img):                                                                 
    h,w = img.shape
    template = np.zeros((h,w))
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if img[height,weight] != 0:
                slice_img = img[height-1:height+2,weight-1:weight+2]
                TF = BoaderOrNot(slice_img)
                if TF:
                    # 1為邊緣
                    template[height-1,weight-1] = 1
                else:
                    # 2為內部
                    template[height - 1, weight - 1] = 2
    return template
def P_or_q(slice_img):
    num = 0
    if slice_img[1,2] ==2:
        num+=1
    if slice_img[0,1] ==2:
        num+=1
    if slice_img[1,0] ==2:
        num+=1
    if slice_img[2,1] ==2:
        num+=1
    return num

def marked(BI_img):

    h,w = BI_img.shape
    BI_img = np.pad(BI_img,(1,1))
    q_p_img = np.zeros((h,w))
    for height in range(1, h + 1):
        for weight in range(1, w + 1):
            if BI_img[height, weight] != 0:
                slice_img = BI_img[height - 1:height + 2, weight - 1:weight + 2]
                num = P_or_q(slice_img)
                if num<1 or BI_img[height, weight] != 1:
                    q_p_img[height-1,weight-1] = 0
                if num>=1 and BI_img[height, weight] == 1:
                    # p點就是3，就是marked的點
                    q_p_img[height-1,weight-1] = 3
    return q_p_img

def count_number(img):
        mask1 = np.array([[1, 1], [0, 0]])
        mask2 = np.array([[1, 0], [1, 0]])
        mask3 = np.array([[0, 0], [1, 1]])
        mask4 = np.array([[0, 1], [0, 1]])
        q = 0
        # 右
        if img[1, 2] != 0:
            slice1 = img[0:2, 1:3]
            num = np.sum(np.bitwise_and(slice1, mask1))
            # print("num",num)
            if num != 2:
                q += 1
        # 上
        if img[0, 1] != 0:
            slice2 = img[0:2, 0:2]
            num = np.sum(np.bitwise_and(slice2, mask2))
            if num != 2:
                q += 1
        # 左
        if img[1, 0] != 0:
            slice3 = img[1:3, 0:2]
            num = np.sum(np.bitwise_and(slice3, mask3))
            if num != 2:
                q += 1
        # 下
        if img[2, 1] != 0:
            slice4 = img[1:3, 1:3]
            num = np.sum(np.bitwise_and(slice4, mask4))
            if num != 2:
                q += 1

        return q

def thinning(original_img,marked_img):
    h,w = original_img.shape
    original_img = np.pad(original_img,(1,1))
    for height in range(1, h + 1):
        for weight in range(1, w + 1):
            slice_img = original_img[height-1:height+2,weight-1:weight+2]
            q_num = count_number(slice_img)
            if q_num==1 and marked_img[height-1,weight-1] == 3:
                original_img[height,weight] = 0
    original_img = original_img[1:h+1,1:w+1]
    return original_img

def recur():
    img = sampling_and_binarize_()
    for _ in range(20):
        img_orginal = img
        img = BI(img)
        marked_img = marked(img)

        img = thinning(img_orginal,marked_img)
        img = np.uint8(img)
        cv2.namedWindow("enhanced", 0);
        cv2.resizeWindow("enhanced", 640, 480);
        cv2.imshow("enhanced", img)
        cv2.waitKey(0)
        # print(_)
        # cv2.imwrite("./lena{}.jpg".format(_),img)
recur()

