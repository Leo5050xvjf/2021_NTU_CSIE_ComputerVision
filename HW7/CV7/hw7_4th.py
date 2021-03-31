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
def yokoi(img):
    # 回傳一個矩陣
    def count_number(img):
        mask1 = np.array([[1, 1], [0, 0]])
        mask2 = np.array([[1, 0], [1, 0]])
        mask3 = np.array([[0, 0], [1, 1]])
        mask4 = np.array([[0, 1], [0, 1]])
        q = 0
        r = 0
        # 右
        if img[1, 2] != 0:
            slice1 = img[0:2, 1:3]
            num = np.sum(np.bitwise_and(slice1, mask1))
            # print("num",num)
            if num != 2:
                q += 1
            else:
                r += 1
        # 上
        if img[0, 1] != 0:
            slice2 = img[0:2, 0:2]
            num = np.sum(np.bitwise_and(slice2, mask2))
            if num != 2:
                q += 1
            else:
                r += 1
        # 左
        if img[1, 0] != 0:
            slice3 = img[1:3, 0:2]
            num = np.sum(np.bitwise_and(slice3, mask3))
            if num != 2:
                q += 1
            else:
                r += 1
        # 下
        if img[2, 1] != 0:
            slice4 = img[1:3, 1:3]
            num = np.sum(np.bitwise_and(slice4, mask4))
            if num != 2:
                q += 1
            else:
                r += 1
        if r == 4:
            return 5
        else:
            return q
    h,w = img.shape
    template = np.zeros((64,64),dtype = int)
    img =np.pad(img,(1,1))
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if img[height,weight] != 0:
                num = count_number(img[height-1:height+2,weight-1:weight+2])
                template[height-1,weight-1] = num
    template = np.uint8(template)
    # template 內是 yokoi number 0~5 組成的影像
    return template

def interior_boarder(img):
    h,w =img.shape
    template = np.zeros((h,w),dtype=int)
    binary_img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    def find_b_i(img_):
        init_ = img_[1,1]
        b_or_not = False
        if img_[1,2] != init_:
            b_or_not = True
        if img_[0, 1] != init_:
            b_or_not = True
        if img_[1,0] != init_:
            b_or_not = True
        if img_[2,1] != init_:
            b_or_not = True
        return b_or_not
    for height in range(1,h+1):
        for weight in range(1, w + 1):
            if binary_img[height,weight] != 0:
                slice_img = binary_img[height-1:height+2,weight-1:weight+2]
                TF =find_b_i(slice_img)
                if TF:
                    # 1為邊界
                    template[height-1,weight-1] = 1
                else:
                    # 2為內部
                    template[height - 1, weight - 1] = 2
    return template

def find_marked(img):
    h,w= img.shape
    template = np.zeros((64,64),dtype=int)
    img = np.pad(img,(1,1))
    for height in range(1,h+1):
        for wwight in range(1,w+1):
            if img[height,wwight] == 2:
                if img[height+1,wwight] == 1:
                    template[height,wwight-1] = 3
                if img[height-1, wwight] == 1:
                    template[height-2,wwight-1] = 3
                if img[height, wwight+1] == 1:
                    template[height-1,wwight] = 3
                if img[height, wwight-1] == 1:
                    template[height-1,wwight-2] = 3
    return template

def count_number_thin(img):
    mask1 = np.array([[1,1],[0,0]])
    mask2 = np.array([[1, 0], [1, 0]])
    mask3 = np.array([[0,0], [1, 1]])
    mask4 = np.array([[0, 1], [0, 1]])
    q = 0
    # 右
    if img[1,2] != 0:
        slice1 = img[0:2,1:3]
        num = np.sum(np.bitwise_and(slice1,mask1))
        # print("num",num)
        if num != 2:q+=1
    # 上
    if img[0,1] != 0:
        slice2 = img[0:2,0:2]
        num = np.sum(np.bitwise_and(slice2,mask2))
        if num != 2:q+=1
    # 左
    if img[1,0] != 0:
        slice3 = img[1:3,0:2]
        num  = np.sum(np.bitwise_and(slice3,mask3))
        if num != 2:q+=1
    # 下
    if img[2,1] != 0:
        slice4 = img[1:3,1:3]
        num  = np.sum(np.bitwise_and(slice4,mask4))
        if num != 2:q+=1
    return q
def thinning(img,marked_img):

    h,w = img.shape
    img = np.pad(img,(1,1))



    for height in range(1,h+1):
        for weight in range(1,w+1):
            if img[height,weight]!=0:
                slice_img = img[height-1:height+2,weight-1:weight+2]
                q = count_number_thin(slice_img)
                print(marked_img[height-1,weight-1])

                if (q == 1) and (marked_img[height-1,weight-1] == 3):
                    print("pppppppppppp")
                    img[height,weight] =0
    print("zzzzzzzzzzzzzzzzzz")
    img = img[1:h+1,1:w+1]
    img = np.uint8(img)
    cv2.namedWindow("enhanced", 0);
    cv2.resizeWindow("enhanced", 640, 480);
    cv2.imshow("enhanced", img)
    cv2.waitKey(0)
    return img

img =sampling_and_binarize_()
for _ in range(100):
    img_or = img
    img = yokoi(img)
    img = interior_boarder(img)
    img = find_marked(img)
    img= thinning(img_or, img)
    print(_)


















