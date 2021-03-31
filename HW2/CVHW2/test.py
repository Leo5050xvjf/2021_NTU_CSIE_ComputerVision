
import cv2
import numpy as np

a = 10
print(a)
counter = 0
img = cv2.imread("./lena.jpg", 0)
# for i in range(10000):
#     for j in range(1000):
#         counter += 1

cv2.imshow("123",img)
cv2.waitKey(0)
