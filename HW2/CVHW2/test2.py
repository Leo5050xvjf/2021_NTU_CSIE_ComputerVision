import cv2
import numpy as np



img = cv2.imread("./lena.jpg", 0).astype('int')
img_counter = np.zeros(img.shape)
h, w = img.shape



# 左上右下的 kernel
kernel_down = np.array([[0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 0]], dtype=np.bool)
kernel_up = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [0, 1, 0]], dtype=np.bool)


# np.pad(圖, ((上, 下), (左, 右)), 方法)
img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')


count = 1
# 填入 counter
for i in range(h):
    for j in range(w):
        if img_pad[i+1, j] != 0 and img_pad[i, j+1] != 0 and img_pad[i+1, j+1] != 0:
            img_counter[i, j] = count
            count += 1

# np.pad(圖, ((上, 下), (左, 右)), 方法)
img_counter_pad = np.pad(img_counter, ((1, 1), (1, 1)), 'constant')



# 上往下
iteration = 5
for iter in range(iteration):  # iteration
    print(f"top down, iter = {iter}")
    for i in range(h):
        for j in range(w):
            cut = img_counter_pad[i:i+3, j:j+3]
            test = cut * kernel_down

            non_zero_pos = np.where(test != 0)
            non_zero_val = test[non_zero_pos]

            for y, x in zip(non_zero_pos[0], non_zero_pos[1]):
                img_counter_pad[i+y, j+x] = np.min(non_zero_val)

    # 下往上
    print(f"bottom up, iter = {iter}")
    for i in range(h-1, -1, -1):
        for j in range(w-1, -1, -1):
            cut = img_counter_pad[i:i + 3, j:j + 3]
            test = cut * kernel_up

            non_zero_pos = np.where(test != 0)
            non_zero_val = test[non_zero_pos]

            for y, x in zip(non_zero_pos[0], non_zero_pos[1]):
                img_counter_pad[i + y, j + x] = np.min(non_zero_val)



index500 = []
# 將 500 以下的填 0
for c in np.unique(img_counter_pad):
    if np.sum(img_counter_pad == c) < 500:
        img_counter_pad[img_counter_pad == c] = 0
    elif c != 0:
        print(f"c = {c}, # = {np.sum(img_counter_pad == c)}")
        index500.append(c)

# 重新編號（為了顯示漂亮）
for num, i in enumerate(index500):
    img_counter_pad[img_counter_pad == i] = num + 1



import matplotlib.pyplot as plt
plt.figure()
plt.imshow(img_counter_pad, cmap='jet')
plt.show()


def mark(img, c):
    x_min = np.min(np.where(img == c)[1]) - 1
    x_max = np.max(np.where(img == c)[1]) - 1
    y_min = np.min(np.where(img == c)[0]) - 1
    y_max = np.max(np.where(img == c)[0]) - 1
    center = (np.mean(np.where(img == c)[1]) - 1, np.mean(np.where(img == c)[0]) - 1)
    return (x_min, y_min), (x_max, y_max), center


def draw_cross(img, center, length = 5):

    x, y = center
    x = int(x)
    y = int(y)
    img[y-1:y+1, x - length: x + length + 1] = np.array([0, 0, 255])
    img[y - length: y + length + 1, x-1:x+1] = np.array([0, 0, 255])

    return img


img = cv2.imread("./lena.jpg")

for i in range(1, len(index500) + 1):
    min_loc, max_loc, center = mark(img_counter_pad, i)
    print(min_loc, max_loc, center)
    cv2.rectangle(img, min_loc, max_loc, (255, 0, 0), 2)
    img = draw_cross(img, center)


cv2.imshow('img', img.astype('uint8'))
cv2.waitKey(0)






