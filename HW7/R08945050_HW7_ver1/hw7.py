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

    cv2.imwrite("./lena_bi.jpg", img)

    return img

def boarder_or_not(slice_img):
    center_ = slice_img[1,1]
    q = 0
    if slice_img[1,2] == center_:
        if slice_img[0, 1] != center_ or slice_img[0, 2] != center_:
            q+=1

    if slice_img[0, 1] == center_:
        if slice_img[0, 0] != center_ or slice_img[1,0] != center_:
            q+=1

    if slice_img[1, 0] == center_:
        if slice_img[2,0] != center_ or slice_img[2,1] != center_:
            q+=1


    if slice_img[2,1] == center_:
        if slice_img[1,2] != center_ or slice_img[2,2] != center_:
            q+=1

    return q

def BI(img):

    h,w = img.shape
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    template = np.zeros((h,w))
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if img [height,weight] != 0:
                slice_im = img[height-1:height+2,weight-1:weight+2]
                q= boarder_or_not(slice_im)
                if q==1:
                    template[height-1,weight-1] = 1
                else:template[height-1,weight-1] = 2
    return template

def marked(BI_img):
    h,w = BI_img.shape
    template = np.zeros((h,w))
    BI_img = np.pad(BI_img,(1,1))
    for height in range(1,h+1):
        for weight in range(1,w+1):
            if BI_img[height,weight] !=0:
                sum_ = 0
                if BI_img[height+1,weight]==1:
                    sum_+=1
                if BI_img[height-1,weight] ==1:
                    sum_+=1
                if BI_img[height,weight+1] == 1:
                    sum_+=1
                if BI_img[height+1,weight-1]==1:
                    sum_+=1

                if sum_<1 or BI_img[height,weight] !=1:
                    template[height-1,weight-1] = 0
                if sum_>=1 and BI_img[height,weight]==1:
                    template[height-1,weight-1] = 1
    return template

def thinning(original_img,marked_img):
    h,w =marked_img.shape
    marked_img = np.pad(marked_img,(1,1))
    original_img =np.pad(original_img,(1,1))


    for height in range(1,h+1):
        for weight in range(1,w+1):
            if marked_img[height,weight] == 1:
                slice_img =original_img[height-1:height+2,weight-1:weight+2]
                q = boarder_or_not(slice_img)
                if q==1:
                    original_img[height,weight] = 0
    original_img=  original_img[1:h+1,1:w+1]
    return original_img


def re():
    img = sampling_and_binarize_()
    h,w = img.shape
    counter =0
    while True:
        img_ori = img
        img = BI(img)
        img = marked(img)
        img  =thinning(img_ori,img)
        img = np.uint8(img)
        img1 = img_ori.reshape((1,h*w))
        img2 = img.reshape((1,h*w))
        cv2.namedWindow("enhanced", 0);
        cv2.resizeWindow("enhanced", 640, 480);
        cv2.imshow("enhanced", img)
        cv2.waitKey(0)
        if (img1 == img2).all() :
            cv2.imwrite("./lena_ans.jpg",img)
            print(counter)
            break
        counter+=1




re()


























