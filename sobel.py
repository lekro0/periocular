import cv2
import numpy as np
from matplotlib import pyplot as plt
import pywt
from scipy import linalg
from cv2 import fastNlMeansDenoising
import os


def adjust_gamma(image, gamma=1.0):
    # 建立查找表，将像素值[0，255]映射到调整后的伽玛值
    # 遍历[0，255]范围内的所有像素值来构建查找表，然后再提高到反伽马的幂-然后将该值存储在表格中
    invGamma = 1.0 / gamma
    image = image.astype(np.uint8)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # 使用查找表应用伽玛校正
    return cv2.LUT(image, table)



def Img_PCA(img):  # img为输入图像，a为保留主成分比例
    #img = adjust_gamma(img, gamma=2.2)

    U, S, V = linalg.svd(img)  # 将img进行奇异值分解，U为左酉矩阵大小M*M，S为奇异值，V为右酉矩阵N*N。
    SS = np.zeros(img.shape)
    a = np.percentile(S, 50)
    if a < 2:
        a = 2
        S[S < a] = a
    for i in range(len(S)):
        SS[i][i] = S[i]  # 将S转化为M*N对角矩阵


    img_pca = np.uint8(np.dot(np.dot(U,SS), V))  # 计算降维处理后的矩阵

    img_pca = adjust_gamma(img_pca, gamma=2.2)
    '''
    scales = [15, 101, 301]
    img_pca = MSR(img_pca, scales)
    #img_pca = adjust_gamma(img_pca)
    '''
    #print(img_pca)
    return img_pca
def sobel(img):
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    #cA = Img_PCA(cA)
    height, width = cH.shape[0], cH.shape[1]

    cH2 = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=1)
    #cH2 = cv2.convertScaleAbs(cH2) #取绝对值，因为会有负数出现造成干扰
    cH2 = cv2.resize(cH2, (int(width), int(height)) , interpolation=cv2.INTER_CUBIC)
    #cH2 = cv2.add(cH, cH2)
    cH2 = cv2.addWeighted(cH, 0.8, cH2, 0.2, 0)  #2是0.9+0.1，3是0.95+0.05，1是0.8+0.2 , 2.5是0.92+0.08
    cV2 = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=1)
    #cV2 = cv2.convertScaleAbs(cV2)
    cV2 = cv2.resize(cV2, (int(width), int(height)) , interpolation=cv2.INTER_CUBIC)
    #cV2 = cv2.add(cV, cV2)
    cV2 = cv2.addWeighted(cV, 0.8, cV2, 0.2, 0)
    #cV2 = tool_Denoising(cV2, VALUE)
    #cV2 = fastNlMeansDenoising(np.uint8(cV2), h = 3, templateWindowSize = 3, searchWindowSize = 5)

    #cD2 = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=1, ksize=3)#duijiao(img, 120)
    #cD2 = cv2.resize(cD2, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
    #cD2 = cv2.addWeighted(cD, 0.8, cD2, 0.2, 0)


    img = pywt.idwt2((cA, (cH2, cV2, cD)), 'haar')
    img = Img_PCA(img)

    img = cv2.bilateralFilter(src=np.uint8(img), d=2, sigmaColor=50, sigmaSpace=15)
    return img






img = sobel(img)

#print(cA2)
#cA = cv2.resize(img , (128, 128), interpolation=cv2.INTER_CUBIC)


#plt.subplot(131),plt.imshow(img, 'gray'), plt.title('cA2')
#plt.subplot(132),plt.imshow(img1, 'gray'), plt.title('img')
plt.imshow(img, 'gray'),plt.axis('off')
#plt.imshow(img, 'gray'), plt.title('result')
plt.show()
