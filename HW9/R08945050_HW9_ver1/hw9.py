import numpy as np
import cv2
from matplotlib import pyplot as plt


def Robert(img,threshold):
    kernel1 = np.array([[1,0],[0,-1]])
    kernel2 = np.array([[0,1],[-1,0]])
    h,w = img.shape
    edge_= np.zeros((h,w),dtype=int)


    for height in range(0,h-1):
        for weight in range(0,w-1):
            slice_img = img[height:height+2,weight:weight+2]
            gx = np.sum(slice_img * kernel1)
            gy = np.sum(slice_img * kernel2)
            g = (gx **2 + gy **2) ** 0.5
            if g >= threshold:
                edge_[height,weight] = 255
            else:
                edge_[height,weight] = 0

    edge_= np.uint8(edge_)
    edge_ = 255-edge_
    return edge_
def Prewitt(img,threshold):
    kernel1 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    kernel2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    h,w = img.shape
    edge_= np.zeros((h,w),dtype=int)
    for height in range(0,h-2):
        for weight in range(0,w-2):
            slice_img = img[height:height+3,weight:weight+3]
            gx = np.sum(slice_img * kernel1)
            gy = np.sum(slice_img * kernel2)
            g = (gx **2 + gy **2) ** 0.5
            if g >= threshold:
                edge_[height,weight] = 255
            else:
                edge_[height,weight] = 0

    edge_= np.uint8(edge_)
    edge_ = 255-edge_
    return edge_
def Sobel(img,threshold):
    kernel1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel2 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    h,w = img.shape
    edge_= np.zeros((h,w))

    for height in range(0,h-2):
        for weight in range(0,w-2):
            slice_img = img[height:height+3,weight:weight+3]
            gx = np.sum(slice_img * kernel1)
            gy = np.sum(slice_img * kernel2)
            g= (gx**2+gy**2) **0.5
            if g >= threshold:
                edge_[height, weight] = 255
            else:
                edge_[height, weight] = 0
    edge_ = np.uint8(edge_)
    edge_= 255-edge_
    return edge_

def FreiAndChen(img,threshold):
    sqrt_2 = 2**0.5
    kernel1 = np.array([[-1,-sqrt_2,-1],[0,0,0],[1,sqrt_2,1]])
    kernel2 = np.array([[-1,0,1],[-sqrt_2,0,sqrt_2],[-1,0,1]])
    h,w = img.shape
    edge_= np.zeros((h,w))
    for height in range(0,h-2):
        for weight in range(0,w-2):
            slice_img = img[height:height+3,weight:weight+3]
            gx = np.sum(slice_img * kernel1)
            gy = np.sum(slice_img * kernel2)
            g = (gx **2 + gy **2) ** 0.5
            if g >= threshold:edge_[height,weight] = 255
            else:edge_[height,weight] = 0
    edge_ = np.uint8(edge_)
    edge_ = 255-edge_
    return edge_
def Kirsch(img,threshold):
    kernel1 =np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    kernel11 =np.rot90(kernel1,2)
    kernel2 =np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    kernel22 =np.rot90(kernel2,2)
    kernel3 =np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    kernel33 =np.rot90(kernel3,2)
    kernel4 =np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    kernel44 =np.rot90(kernel4,2)
    kernel_list = [kernel1, kernel11, kernel2, kernel22, kernel3, kernel33, kernel4, kernel44]
    h, w = img.shape
    Gradient = np.zeros((h, w))

    for height in range(0,h-2):
        for weight in range(0,w-2):
            slice_img = img[height:height+3,weight:weight+3]
            compare_ = []
            for _ in kernel_list:
                gx = np.sum(slice_img * _)
                compare_.append(gx)
            max_gx = max(compare_)
            if max_gx >= threshold:Gradient[height,weight] = 255
            else:Gradient[height, weight] = 0
    Gradient = np.uint8(Gradient)
    Gradient = 255-Gradient
    return Gradient
def Robinson(img,threshold):
    h,w = img.shape
    East = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    West = np.rot90(East,2)
    Northeast = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    Southwest =np.rot90(Northeast,2)
    North =np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    South =np.rot90(North,2)
    Northwest =np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
    Southeast = np.rot90(Northwest,2)
    directions  = [ East ,West, Northeast ,Southwest,North ,South ,Northwest ,Southeast]
    Gradient = np.zeros((h, w))
    for height in range(0,h-2):
        for weight in range(0,w-2):
            slice_img = img[height:height + 3, weight:weight + 3]
            compare_ = []
            for _ in directions:
                gx = np.sum(slice_img * _)
                compare_.append(gx)
            max_gx = max(compare_)
            if max_gx >= threshold:Gradient[height, weight] = 255
            else:Gradient[height, weight] = 0
    Gradient = np.uint8(Gradient)
    Gradient = 255-Gradient
    return Gradient

def Nevatia_Babu(img,threshold):
    kernel1 = np.array([[100,100,100,100,100],
                        [100,100,100,100,100],
                        [0,0,0,0,0],
                        [-100,-100,-100,-100,-100],
                        [-100,-100,-100,-100,-100]])

    kernel2 = np.array([[100,100,100,100,100],
                        [100,100,100,78,-32],
                        [100,92,0,-92,-100],
                        [32,-78,-100,-100,-100],
                        [-100,-100,-100,-100,-100]])

    kernel3 = np.array([[100,100,100,32,-100],
                        [100,100,92,-78,-100],
                        [100,100,0,-100,-100],
                        [100,78,-92,-100,-100],
                        [100,-32,-100,-100,-100]])

    kernel4 = np.array([[-100,-100,0,100,100],
                        [-100,-100,0,100,100],
                        [-100,-100,0,100,100],
                        [-100,-100,0,100,100],
                        [-100,-100,0,100,100]])

    kernel5 = np.array([[-100,-100,0,100,100],
                        [-100,-100,0,100,100],
                        [-100,-100,0,100,100],
                        [-100,-100,0,100,100],
                        [-100,-100,0,100,100]])

    kernel6 = np.array([[100,100,100,100,100],
                        [-32,78,100,100,100],
                        [-100,-92,0,92,100],
                        [-100,-100,-100,-78,32],
                        [-100,-100,-100,-100,-100]])
    kernel_list = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
    h,w = img.shape
    Gradient = np.zeros((h, w))
    for height in range(0,h-4):
        for weight in range(0,w-4):
            slice_img = img[height:height+5,weight:weight+5]
            compare_ = []
            for _ in kernel_list:
                gx = np.sum(slice_img * _)
                compare_.append(gx)
            max_gx = max(compare_)
            if max_gx >= threshold:Gradient[height, weight] = 255
            else:Gradient[height, weight] = 0

    Gradient = np.uint8(Gradient)
    Gradient = 255-Gradient
    return Gradient





if __name__ == "__main__":
    img = cv2.imread("./lena.bmp", 0)
    img1 = Robert(img,12)
    img2 = Prewitt(img,24)
    img3 = Sobel(img,38)
    img4 = FreiAndChen(img,30)
    img5 = Kirsch(img,135)
    img6 =Robinson(img,43)
    img7 =Nevatia_Babu(img,12500)
    img_name = ["Robert","Prewitt","Sobel","FreiAndChen","Kirsch"," Robinson"," Nevatia_Babu"]
    img_list = [img1,img2,img3,img4,img5,img6,img7]
    for _ in zip(img_name,img_list):
        cv2.imwrite("./{}.jpg".format(_[0]),_[1])
    for _ in zip(img_name,img_list):
        cv2.imshow(_[0],_[1])
        cv2.waitKey(0)
