import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(img):
    cv2.imshow("test",img)
    cv2.waitKey(0)
def write(img,name):
    cv2.imwrite("./"+name,img)

def histogram():
    img = cv2.imread("./lena.bmp", cv2.IMREAD_GRAYSCALE)
    w,h = img.shape
    dict1 = {}
    for i in range(255):
        dict1[i] = 0
    for row in range(w):
        for col in range(h):
            dict1[img [row,col]] += 1
    X = [i for i in range(255)]
    x = [pixel for pixel in dict1.values()]
    plt.title('Lena Histogram')
    plt.xlabel("Bin")
    plt.ylabel("# of pixel")
    plt.bar(X, x, alpha=0.5, width=1, facecolor='red', edgecolor='black', label='two', lw=1)
    plt.show()

def histogram_3():
    img = cv2.imread("./lena.bmp", cv2.IMREAD_GRAYSCALE)
    write(img,"01.jpg")
    img = img//3
    write(img,"02.jpg")
    show(img)
    w,h = img.shape
    dict1 = {}
    for i in range(255):
        dict1[i] = 0
    for row in range(w):
        for col in range(h):
            dict1[img [row,col]] += 1
    X = [i for i in range(255)]
    x = [pixel for pixel in dict1.values()]
    plt.title('Lena Histogram')
    plt.xlabel("Bin")
    plt.ylabel("# of pixel")
    plt.bar(X, x, alpha=0.5, width=1, facecolor='red', edgecolor='black', label='two', lw=1)
    plt.show()

def trans_his_to_cdf():
    img = cv2.imread("./lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = img // 3
    w, h = img.shape
    show(img)
    original_list = [0 for i in range(256)]
    for i in range(h):
        for j in range(w):
            original_list[img[i,j]] +=1
    cum_list = np.cumsum(original_list)
    min_cum = min(cum_list)
    traslated_cum_list = []
    for cdf_val in cum_list:
        cdf = round(((cdf_val-min_cum)/(512*512-min_cum))*(255))
        traslated_cum_list.append(cdf)
    look_up_table = {}
    for i in range(256):
        look_up_table[i] = traslated_cum_list[i]

    for i in range(h):
        for j in range(w):
            img[i,j] = look_up_table[img[i,j]]
    dict1 = {}
    for i in range(256):
        dict1[i] = 0
    for row in range(h):
        for col in range(w):
            dict1[img [row,col]] += 1
    X = [i for i in range(255)]
    x = [pixel for pixel in dict1.values()]
    plt.title('Lena Histogram_CDF')
    plt.xlabel("Bin")
    plt.ylabel("# of pixel")
    plt.bar(X, x, alpha=0.5, width=1, facecolor='red', edgecolor='black', label='two', lw=1)
    plt.show()
    show(img)
if __name__ == "__main":
    histogram()
    histogram_3()
    trans_his_to_cdf()









