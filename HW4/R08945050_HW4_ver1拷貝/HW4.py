import numpy as np
import cv2


def lena_binary():
    pass

def mask(N):
    mask = np.ones((N,N))
    mask[0,0] = 0
    mask[0,N-1] = 0
    mask[N-1,N-1] = 0
    mask[N-1,0] = 0
    return mask
# 侵蝕
def erosion():
    img = cv2.imread("./lena.jpg",cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    for height in range(h):
        for width in range(w):
            if img[height,width]>=128:
                img[height,width] = 255
            else:
                img[height, width] = 0
    img_pad = np.pad(img,(2,2),'constant')
    img_pad =img_pad/255
    masks = mask(5)
    temp_img = np.zeros((h,w))
    for height in range(2,h+2):
        for width in range(2,w+2):
            slice_pad = img_pad[height-2:height+3,width-2:width+3]
            # 21 = 3+5+5+5+3=>八角形的mask
            if np.sum(cv2.bitwise_and(slice_pad,masks)) == 21:
                temp_img[height-2,width-2] = 255
    temp_img = np.uint8(temp_img)
    cv2.imwrite("./lena_erosion.jpg",temp_img)
    return 0


# 膨脹
def dilation():
    img = cv2.imread("./lena.jpg",cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    for height in range(h):
        for width in range(w):
            if img[height,width]>=128:
                img[height,width] = 255
            else:
                img[height, width] = 0

    # img = denoise(img)
    img_pad = np.pad(img,(2,2),'constant')
    img_pad =img_pad/255
    masks = mask(5)
    temp_img = np.zeros((h+4,w+4))
    for height in range(2,h+2):
        for width in range(2,w+2):
            if img_pad[height,width] != 0:
                slice_pad = img_pad[height-2:height+3,width-2:width+3]
                slice_pad_and_masks_or =  cv2.bitwise_or(slice_pad,masks)
                final_ans = cv2.bitwise_or(slice_pad_and_masks_or,temp_img[height-2:height+3,width-2:width+3])
                temp_img[height-2:height+3,width-2:width+3] = final_ans

    temp_img = (np.uint8(temp_img))*255
    cv2.imwrite("./lena_dilation.jpg",temp_img)
    return 0



'''如何把 binary的圖 黑與白 反轉？（不確定是否有Func()）

    ans:建立一個等大的255影像，將兩個影線相減，例如： 255影像 - 欲黑白翻轉的影像 = 結果
'''
def opening():
    '''A。B
    幾何意義為：“所有” “可完全放進A中的B” 的聯集'''
    img = cv2.imread("./lena.jpg",cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    for height in range(h):
        for width in range(w):
            if img[height,width]>=128:
                img[height,width] = 255
            else:
                img[height, width] = 0
    masks = np.uint(mask(5))
    img_pad = np.pad(img, (2, 2), 'constant')
    temp_img =np.zeros((h+4,w+4))
    for height in range(2,h+2):
        for width in range(2,w+2):
            slice_pad = img_pad[height-2:height+3,width-2:width+3]
            if np.sum(np.bitwise_and(slice_pad,masks)) == 21:
                temp_img[height,width] = 255
    '''做完erotion了 剩下dilasion'''
    ans = np.uint(np.zeros((h+4,w+4)))
    temp_img = np.uint8(temp_img)//255

    for height in range(2,h+2):
        for width in range(2,w+2):
            if temp_img[height,width] != 0:
                slice_pad = temp_img[height-2:height+3,width-2:width+3]
                slice_pad_and_masks_or =  np.bitwise_or(slice_pad,masks)
                final_ans = np.bitwise_or(slice_pad_and_masks_or,ans[height-2:height+3,width-2:width+3])
                ans[height-2:height+3,width-2:width+3] = final_ans
    ans = np.uint8(ans)
    ans =ans*255
    cv2.imwrite("./lena_opening.jpg", ans)




def closing():
    '''A·B
    幾何意義為：”不和A重疊之所有B平移“的聯集“的補集”
    流程：1.先把A取補集，
         2.在對A的補集做B的erosion，得到一個“較胖”的A，且A的值為0，
         3.所以要再做一次補集
    '''

    img = cv2.imread("./lena.jpg",cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    for height in range(h):
        for width in range(w):
            if img[height,width]>=128:
                img[height,width] = 255
            else:
                img[height, width] = 0
    img = np.pad(img,(2,2))
    img_complement = 255-img
    masks = np.uint(mask(5))
    temp_img = np.uint(np.zeros((h + 4, w + 4)))
    for i in range(2,h+2):
        for j in range(2,w+2):
            if np.sum(np.bitwise_and(masks,img_complement[i-2:i+3,j-2:j+3])) == 21:
                final_bitwise = np.bitwise_or(masks,img_complement[i-2:i+3,j-2:j+3])
                temp_img[i-2:i+3,j-2:j+3] = final_bitwise
    temp_img = np.uint8(temp_img)
    temp_img = 255-temp_img
    cv2.imwrite("./lena_closing.jpg", temp_img)



def hit_and_miss():
    img = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    for height in range(h):
        for width in range(w):
            if img[height, width] >= 128:
                img[height, width] = 1
            else:
                img[height, width] = 0
    img = np.pad(img,(1,1),'constant')
    temp_img = np.zeros((h+2,w+2))

    img_complement = 1-img
    temp_img_com = np.ones((h+2,w+2))

    mask_A = [[0,0,0],[1,1,0],[0,1,0]]
    mask_A = np.array(mask_A)

    mask_AC=[[0,1,1],[0,0,1],[0,0,0]]
    mask_AC = np.array(mask_AC)


    for i in range(1,h+1):
        for j in range(1,w+1):
            if img[i,j] != 0:
                slice_pad = img[i-1:i+2,j-1:j+2]
                if np.sum(np.bitwise_and(slice_pad,mask_A)) == 3:
                    temp_img[i,j] = 1

    for i in range(1,h+1):
        for j in range(1,w+1):
            slice_pad = img_complement[i-1:i+2,j-1:j+2]
            if np.sum(np.bitwise_and(slice_pad,mask_AC)) != 3:
                temp_img_com[i,j] = 0
    temp_img_com =np.uint(temp_img_com)
    temp_img = np.uint(temp_img)
    ans = cv2.bitwise_and(temp_img_com,temp_img)
    ans = np.uint8(ans*255)
    cv2.imwrite("./lena_hit_and_miss.jpg",ans)


if __name__ == "__main__":
    erosion()
    dilation()
    hit_and_miss()
    opening()
    closing()