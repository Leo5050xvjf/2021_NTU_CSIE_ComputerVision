import  numpy as np
import cv2
import  random
import math
def GaussianNoise(img,amp):
    h,w = img.shape
    noise = np.random.randn(h,w)*amp
    noise_img =img+noise
    noise_img = np.uint8(noise_img)
    return noise_img
def SaltAndPepper(img,probability):
    h,w = img.shape
    salt_pepper = np.zeros((h,w))
    for height in range(h):
        for weight in range(w):
            randomValue = random.uniform(0, 1)
            if (randomValue <= probability):
                salt_pepper[height,weight] = 0
            elif (randomValue >= 1 - probability):
                salt_pepper[height,weight] = 255
            else:
                salt_pepper[height,weight] = img[height,weight]
    salt_pepper = np.uint8(salt_pepper)
    return salt_pepper
def BoxFilter(img,kernel_size):
    kernel   =  np.ones((kernel_size,kernel_size),dtype=int)/(kernel_size**2)
    h,w = img.shape
    img = np.pad(img,((kernel_size-1)//2,(kernel_size-1)//2))
    template = np.zeros((h,w),dtype=int)
    slice_range = [((kernel_size-1)//2),((kernel_size-1)//2)+1]
    for height in range(slice_range[0],h+slice_range[0]):
        for weight in range(slice_range[0],w+slice_range[0]):
            slice_img = img[height-slice_range[0]:height+slice_range[1],weight-slice_range[0]:weight+slice_range[1]]
            sum_  = np.sum(kernel * slice_img)
            template[height-(slice_range[0]),weight- slice_range[0]] = sum_
    template = np.uint8(template)
    return template
def MedianFilter(img,kernel_size):

    h,w = img.shape
    # kernel = np.zeros((kernel_size, kernel_size), dtype=int) / (kernel_size ** 2)
    img = np.pad(img,((kernel_size-1)//2,(kernel_size-1)//2))
    template = np.zeros((h,w),dtype=int)
    slice_range = [((kernel_size-1)//2),((kernel_size-1)//2)+1]
    for height in range(slice_range[0],h+slice_range[0]):
        for weight in range(slice_range[0],w+slice_range[0]):
            slice_img = img[height-slice_range[0]:height+slice_range[ 1],weight-slice_range[0]:weight+slice_range[1]]
            one_D= np.reshape(slice_img,(kernel_size**2))
            one_D = np.sort(one_D)
            template[height - (slice_range[0]), weight - slice_range[0]] = one_D[(kernel_size**2-1)//2]
    template = np.uint8(template)

    return template
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
    return temp_
def opening(img):
    e = erosion(img)
    d = dilation_(e)
    return d
def closing(img):
    d = dilation_(img)
    e = erosion(d)
    return e
def SNR(img_ori,img_noise):
    img_ori = img_ori/255
    h,w = img_ori.shape
    img_noise = img_noise/255
    mu_s = (np.sum(img_ori))/(h*w)
    VS =(np.sum((img_ori-mu_s)**2))/(h*w)
    mu_noise = np.sum(img_noise-img_ori)/(h*w)
    VN = (np.sum((img_noise-img_ori-mu_noise)**2))/(h*w)
    x= (VS/VN)**0.5
    return 20*math.log(x,10)

if __name__ == "__main__":
    img = cv2.imread("./lena.bmp",0)
    gaussiannoise10  = GaussianNoise(img,10)
    cv2.imwrite("./cvhw8/noise_img/gaussiannoise10 .jpg",gaussiannoise10)
    gaussiannoise30  = GaussianNoise(img,30)
    cv2.imwrite("./cvhw8/noise_img/gaussiannoise30 .jpg",gaussiannoise30)
    saltpepper005 = SaltAndPepper(img,0.05)
    cv2.imwrite("./cvhw8/noise_img/saltpepper005.jpg",saltpepper005)
    saltpepper010 = SaltAndPepper(img,0.1)
    cv2.imwrite("./cvhw8/noise_img/saltpepper010.jpg",saltpepper010)


    # SNR
    SNR1 = SNR(img,gaussiannoise10)
    print("SNR_gaussiannoise10",SNR1)
    SNR2 = SNR(img,gaussiannoise30)
    print("SNR_gaussiannoise30",SNR2)
    SNR3 = SNR(img,saltpepper005)
    print("SNR_saltpepper005",SNR3)
    SNR4 = SNR(img,saltpepper010)
    print("SNR_saltpepper010",SNR4)
    # box filte *8

    box1 = BoxFilter(gaussiannoise10,3)
    cv2.imwrite("./cvhw8/box_filter/box1.jpg",box1)
    box2 = BoxFilter(gaussiannoise10,5)
    cv2.imwrite("./cvhw8/box_filter/box2.jpg",box2)
    box3 = BoxFilter(gaussiannoise30,3)
    cv2.imwrite("./cvhw8/box_filter/box3.jpg",box3)
    box4 = BoxFilter(gaussiannoise30,5)
    cv2.imwrite("./cvhw8/box_filter/box4.jpg",box4)
    box5 = BoxFilter(saltpepper005,3)
    cv2.imwrite("./cvhw8/box_filter/box5.jpg",box5)
    box6 = BoxFilter(saltpepper010,5)
    cv2.imwrite("./cvhw8/box_filter/box6.jpg",box6)
    box7 = BoxFilter(saltpepper005,5)
    cv2.imwrite("./cvhw8/box_filter/box7.jpg",box7)
    box8 = BoxFilter(saltpepper010,3)
    cv2.imwrite("./cvhw8/box_filter/box8.jpg",box8)

    SNR_box1 = SNR(img,box1)
    print("box_gaussiannoise10,3",SNR_box1)
    SNR_box2 = SNR(img,box2)
    print("box_gaussiannoise10,5",SNR_box2)
    SNR_box3 = SNR(img,box3)
    print("box_gaussiannoise30,3",SNR_box3)
    SNR_box4 = SNR(img,box4)
    print("box_gaussiannoise30,5",SNR_box4)
    SNR_box5 = SNR(img,box5)
    print("box_saltpepper005,3",SNR_box5)
    SNR_box6 = SNR(img,box6)
    print("box_saltpepper010,5",SNR_box6)
    SNR_box7 = SNR(img,box7)
    print("box_saltpepper005,5",SNR_box7)
    SNR_box8 = SNR(img,box8)
    print("box_saltpepper010,3",SNR_box8)
    # median filter *8

    median1 =MedianFilter(gaussiannoise10,3)
    cv2.imwrite("./cvhw8/median_filter/median1.jpg",median1)
    median2 =MedianFilter(gaussiannoise30,3)
    cv2.imwrite("./cvhw8/median_filter/median2.jpg",median2)
    median3 =MedianFilter(gaussiannoise10,5)
    cv2.imwrite("./cvhw8/median_filter/median3.jpg",median3)
    median4 =MedianFilter(gaussiannoise30,5)
    cv2.imwrite("./cvhw8/median_filter/median4.jpg",median4)
    median5 =MedianFilter(saltpepper005,3)
    cv2.imwrite("./cvhw8/median_filter/median5.jpg",median5)
    median6 =MedianFilter(saltpepper005,5)
    cv2.imwrite("./cvhw8/median_filter/median6.jpg",median6)
    median7 =MedianFilter(saltpepper010,3)
    cv2.imwrite("./cvhw8/median_filter/median7.jpg",median7)
    median8 =MedianFilter(saltpepper010,5)
    cv2.imwrite("./cvhw8/median_filter/median8.jpg",median8)
    SNR_median1 = SNR(img,median1 )
    print("median_gaussiannoise10,3",SNR_median1)
    SNR_median2 = SNR(img,median2)
    print("median_gaussiannoise30,3",SNR_median2)
    SNR_median3  = SNR(img,median3 )
    print("mediangaussiannoise10,5",SNR_median3)
    SNR_median4 = SNR(img,median4 )
    print("median_gaussiannoise30,5",SNR_median4)
    SNR_median5 = SNR(img,median5 )
    print("median_saltpepper005,3",SNR_median5)
    SNR_median6 = SNR(img,median6 )
    print("median_saltpepper005,5",SNR_median6)
    SNR_median7 = SNR(img,median7 )
    print("median_saltpepper010,3",SNR_median7)
    SNR_median8 = SNR(img,median8 )
    print("median_saltpepper010,5",SNR_median8)
    # opening closing
    d1= opening(gaussiannoise10)
    cv2.imwrite("./cvhw8/opening/d1.jpg",d1)
    d2= opening(gaussiannoise30)
    cv2.imwrite("./cvhw8/opening/d2.jpg",d2)
    d3=opening(saltpepper010)
    cv2.imwrite("./cvhw8/opening/d3.jpg",d3)
    d4 = opening(saltpepper005)
    cv2.imwrite("./cvhw8/opening/d4.jpg",d4)

    d1_SNR = SNR(img,d1)
    print("opening gaussiannoise10_SNR",d1_SNR)
    d2_SNR = SNR(img,d2)
    print("opening gaussiannoise30_NR",d2_SNR)
    d3_SNR =SNR(img,d3)
    print("opening saltpepper010_SNR",d3_SNR)
    d4_SNR = SNR(img,d4)
    print("opening saltpepper005_SNR",d4_SNR)



    e1= closing(gaussiannoise10)
    cv2.imwrite("./cvhw8/closing/e1.jpg",e1)
    e2= closing(gaussiannoise30)
    cv2.imwrite("./cvhw8/closing/e2.jpg",e2)
    e3=closing(saltpepper005)
    cv2.imwrite("./cvhw8/closing/e3.jpg",e3)
    e4=closing(saltpepper010)
    cv2.imwrite("./cvhw8/closing/e4.jpg",e4)

    e1_SNR = SNR(img,e1)
    print("closing gaussiannoise10_SNR",e1_SNR)
    e2_SNR = SNR(img,e2)
    print("closing gaussiannoise30_SNR",e2_SNR)
    e3_SNR = SNR(img,e3)
    print("closing saltpepper005_SNR",e3_SNR)
    e4_SNR = SNR(img,e4)
    print("closing saltpepper010_SNR",e4_SNR)

































