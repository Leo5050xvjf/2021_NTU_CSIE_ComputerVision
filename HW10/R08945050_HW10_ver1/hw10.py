
import cv2
import numpy as np



def Laplacian(img,threshold1,threshold2):
    mask1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    mask2 = np.array([[1,1,1],[1,-8,1],[1,1,1]]) *(1/3)
    mask_list = [mask1,mask2]
    threshold_ = [threshold1,threshold2]
    h,w =img.shape
    template1= np.zeros((h,w),dtype=int)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    img_list = []
    for _ in zip(mask_list,threshold_):
        for height in range(1,h+1):
            for weight in range(1,w+1):
                slice_img = img[height-1:height+2,weight-1:weight+2]
                val = np.sum(slice_img * _[0])
                if val >= _[1]:template1[height-1,weight-1] = 1
                elif val <= -_[1] : template1[height-1,weight-1] = -1
                else:template1[height-1,weight-1] = 0
        #template1 是充滿1,0,-1的影像
        template1 = np.pad(template1,(1,1))
        img_list.append(template1)
        template1= np.zeros((h,w),dtype=int)
    edge_img = np.zeros((h,w),dtype=int)
    edge_img_list = []
    for img_ in img_list:
        for height in range(1,h+1):
            for weight in range(1,w+1):
                if img_[height,weight] == 1:
                    slice_img_ = img_[height-1:height+2,weight-1:weight+2]
                    one_D   = np.reshape(slice_img_,(1,9))
                    if -1 in one_D:edge_img[height-1,weight-1] = 0
                    else:edge_img[height-1,weight-1] = 255
                else:edge_img[height-1,weight-1] = 255
        edge_img = np.uint8(edge_img)
        edge_img_list.append(edge_img)
        edge_img = np.zeros((h,w),dtype=int)
    for _ in enumerate(edge_img_list):
        cv2.imwrite("./Laplacian{}.jpg".format(_[0]),_[1])
def Minimum_variance_Laplacian(img,threshold):
    mask = np.array([[2,-1,2],[-1,-4,-1],[2,-1,2]]) * (1/3)
    h,w =img.shape
    template1= np.zeros((h,w),dtype=int)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    for height in range(1, h + 1):
        for weight in range(1, w + 1):
            slice_img = img[height - 1:height + 2, weight - 1:weight + 2]
            val = np.sum(slice_img * mask)
            if val >= threshold:
                template1[height - 1, weight - 1] = 1
            elif val <= -threshold:
                template1[height - 1, weight - 1] = -1
            else:
                template1[height - 1, weight - 1] = 0
    edge_img = np.zeros((h, w), dtype=int)
    template1 = np.pad(template1,(1,1))
    for height in range(1, h + 1):
        for weight in range(1, w + 1):
            if template1[height, weight] == 1:
                slice_img_ = template1[height - 1:height + 2, weight - 1:weight + 2]
                one_D = np.reshape(slice_img_, (1, 9))
                if -1 in one_D:
                    edge_img[height - 1, weight - 1] = 0
                else:
                    edge_img[height - 1, weight - 1] = 255
            else:
                edge_img[height - 1, weight - 1] = 255
    edge_img = np.uint8(edge_img)
    cv2.imwrite("./Minimum_variance_Laplacian.jpg",edge_img)
def LOG(img,threshold):
    mask = np.array([[0,0,0,-1,-1,-2,-1,-1,0,0,0],
            [0,0,-2,-4,-8,-9,-8,-4,-2,0,0],
            [0,-2,-7,-15,-22,-23,-22,-15,-7,-2,0],
            [-1,-4,-15,-24,-14,-1,-14,-24,-15,-4,-1],
           [-1,-8,-22,-14,52,103,52,-14,-22,-8,-1],
            [-2,-9,-23,-1,103,178,103,-1,-23,-9,-2],
            [-1,-8,-22,-14,52,103,52,-14,-22,-8,-1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0,-2,-7,-15,-22,-23,-22,-15,-7,-2,0],
            [0,0,-2,-4,-8,-9,-8,-4,-2,0,0],
            [0,0,0,-1,-1,-2,-1,-1,0,0,0]])
    h,w =img.shape
    template1= np.zeros((h,w),dtype=int)
    img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REFLECT)
    for height in range(5, h + 5):
        for weight in range(5, w + 5):
            slice_img = img[height - 5:height + 6, weight - 5:weight + 6]
            val = np.sum(slice_img * mask)
            if val >= threshold:
                template1[height - 5, weight - 5] = 1
            elif val <= -threshold:
                template1[height - 5, weight - 5] = -1
            else:
                template1[height - 5, weight - 5] = 0
    edge_img = np.zeros((h, w), dtype=int)
    template1 = np.pad(template1,(1,1))
    for height in range(1, h + 1):
        for weight in range(1, w + 1):
            if template1[height, weight] == 1:
                slice_img_ = template1[height - 1:height + 2, weight - 1:weight + 2]
                one_D = np.reshape(slice_img_, (1, 9))
                if -1 in one_D:
                    edge_img[height - 1, weight - 1] = 0
                else:
                    edge_img[height - 1, weight - 1] = 255
            else:
                edge_img[height - 1, weight - 1] = 255
    edge_img = np.uint8(edge_img)
    cv2.imwrite("./LOG.jpg", edge_img)


def DOG(img,threshold):
    mask = np.array([[-1,-3,-4,-6,-7,-8,-7,-6,-4,-3,-1],
            [-3,-5,-8,-11,-13,-13,-13,-11,-8,-5,-3],
            [-4,-8,-12,-16,-17,-17,-17,-16,-12,-8,-4],
            [-6,-11,-16,-16,0,15,0,-16,-16,-11,-6],
            [-7,-13,-17,0,85,160,85,0,-17,-13,-7],
            [-8,-13,-17,15,160,283,160,15,-17,-13,-8],
            [-7,-13,-17,0,85,160,85,0,-17,-13,-7],
            [-6,-11,-16,-16,0,15,0,-16,-16,-11,-6],
            [-4,-8,-12,-16,-17,-17,-17,-16,-12,-8,-4],
            [-3,-5,-8,-11,-13,-13,-13,-11,-8,-5,-3],
            [-1,-3,-4,-6,-7,-8,-7,-6,-4,-3,-1],
            ])
    h, w = img.shape
    template1 = np.zeros((h, w), dtype=int)
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REFLECT)
    for height in range(5, h + 5):
        for weight in range(5, w + 5):
            slice_img = img[height - 5:height + 6, weight - 5:weight + 6]
            val = np.sum(slice_img * mask)
            if val >= threshold:
                template1[height - 5, weight - 5] = 1
            elif val <= -threshold:
                template1[height - 5, weight - 5] = -1
            else:
                template1[height - 5, weight - 5] = 0
    edge_img = np.zeros((h, w), dtype=int)
    template1 = np.pad(template1, (1, 1))
    for height in range(1, h + 1):
        for weight in range(1, w + 1):
            if template1[height, weight] == 1:
                slice_img_ = template1[height - 1:height + 2, weight - 1:weight + 2]
                one_D = np.reshape(slice_img_, (1, 9))
                if -1 in one_D:
                    edge_img[height - 1, weight - 1] = 0
                else:
                    edge_img[height - 1, weight - 1] = 255
            else:
                edge_img[height - 1, weight - 1] = 255
    edge_img = np.uint8(edge_img)
    cv2.imwrite("./DOG.jpg", edge_img)
if __name__ == "__main__":
    img = cv2.imread("./lena.bmp", 0)
    Laplacian(img, 15, 15)
    Minimum_variance_Laplacian(img, 20)
    LOG(img, 3000)
    DOG(img, 1)



