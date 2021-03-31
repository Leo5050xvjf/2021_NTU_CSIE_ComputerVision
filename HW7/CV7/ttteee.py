
import cv2
import  numpy as np

# img = cv2.imread("./lena.bmp", 0)
img = np.array([[1,2,3],
                [2,3,4],
                [4,5,6]])
# img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
# print(img)
img1 = img.reshape(1,9)
print(img1)



