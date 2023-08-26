import cv2
import numpy as np
import os
import math
#encoding=utf-8
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# brenner梯度函数计算
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 2):
        for y in range(0, shape[1]):
            out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
    return out/(shape[0]*shape[1])

# def tenengrad(img):
#     sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=3)
#     sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#     gradient = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
#     score = np.mean((gradient))
#     return score


# Laplacian梯度函数计算
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    return cv2.Laplacian(img, cv2.CV_64F).var()


# SMD梯度函数计算
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0] - 1):
        for y in range(0, shape[1]):
            out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
            out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
    return out/(shape[0]*shape[1])


# SMD2梯度函数计算
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(int(img[x, y] - int(img[x, y + 1])))
    return out/(shape[0]*shape[1])


# 方差函数计算
def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            out += (img[x, y] - u) ** 2
    return out/(shape[0]*shape[1])


# energy函数计算
def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
    return out/(shape[0]*shape[1])


# Vollath函数计算
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0] * shape[1] * (u ** 2)
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1]):
            out += int(img[x, y]) * int(img[x + 1, y])
    return out/(shape[0]*shape[1])


# entropy函数计算
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0] * np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i] != 0:
            out -= p[i] * math.log(p[i] / count) / count
    return out


def main(img1):
    # print('Brenner', brenner(img1))
    # print('Laplacian', Laplacian(img1))
    # print('SMD', SMD(img1))
    # print('SMD2', SMD2(img1))
    # print('Variance', variance(img1))
    # print('Energy', energy(img1))
    # print('Vollath', Vollath(img1))
    # print('Entropy', entropy(img1))
    entroy_value = entropy(img1)
    # NRSS_value = NRSS(img1)
    return entroy_value

def gauseBlur(img):
    img_Guassian = cv2.GaussianBlur(img,(7,7),0)
    return img_Guassian

def loadImage(filepath):
    img = cv2.imread(filepath, 0)  ##   读入灰度图
    return img

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

def saveImage(path, img):
    cv2.imwrite(path, img)

def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

def getBlock(G,Gr):
    (h, w) = G.shape
    G_blk_list = []
    Gr_blk_list = []
    sp = 6
    for i in range(sp):
        for j in range(sp):
            G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
            G_blk_list.append(G_blk)
            Gr_blk_list.append(Gr_blk)
    sum = 0
    for i in range(sp*sp):
        mssim = compare_ssim(G_blk_list[i], Gr_blk_list[i])
        sum = mssim + sum
    nrss = 1-sum/(sp*sp*1.0)
    # print(nrss)
    return nrss

def NRSS(path):
    image = loadImage(path)
    #高斯滤波
    Ir = gauseBlur(image)
    G = sobel(image)
    Gr = sobel(Ir)
    blocksize = 8
    value = getBlock(G, Gr)
    ## 获取块信息
    return value

if __name__ == '__main__':
    file_path = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/4TB/segmentation_super/non-reference metric_3'
    file_list = os.listdir(file_path)
    file_list.sort()
    for item0 in file_list:
        file_path1 = os.path.join(file_path,item0)
        file_list1 = os.listdir(file_path1)
        file_list1.sort()
        for item1 in file_list1:
            file_path2 = os.path.join(file_path1,item1)
            file_list2 = os.listdir(file_path2)
            file_list2.sort()
            entroy_va = []
            NRSS_va = []
            for item in file_list2:
                image_path = os.path.join(file_path2, item)
                img = cv2.imread(image_path, 0)
                # img = cv2.resize(img,(304,304))
                # img = cv2.resize(img, (512, 512))
                entroy_value = main(img)
                # value = NRSS(image_path)
                value = NRSS(image_path)
                # print(NRSS_value)
                entroy_va.append(entroy_value)
                NRSS_va.append(value)
            print(item0+item1, str(np.array(entroy_va).mean()), str(np.array(entroy_va).std()), str(np.array(NRSS_va).mean()),
                  str(np.array(NRSS_va).std()))



    # 读入原始图像
    # file_total  = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/domain_adaptation_1/haohy_domain_ada_sr_optovue+sr_domain_ada_new2'
    # file_total_list = os.listdir(file_total)
    # file_total_list.sort()
    # i=0
    # for item0 in file_total_list:
    #     file_path = os.path.join(file_total,item0)
    #     file_list = os.listdir(file_path)
    #     file_list.sort()
    #     entroy_va = []
    #     NRSS_va = []
    #     for item in file_list:
    #         image_path = os.path.join(file_path, item)
    #         img = cv2.imread(image_path, 0)
    #         img = cv2.resize(img,(304,304))
    #         # img = cv2.resize(img, (512, 512))
    #         entroy_value = main(img)
    #         # value = NRSS(image_path)
    #         value = brenner(img)
    #         # print(NRSS_value)
    #         entroy_va.append(entroy_value)
    #         NRSS_va.append(value)
    #     print(i,str(np.array(entroy_va).mean()), str(np.array(entroy_va).std()), str(np.array(NRSS_va).mean()),
    #           str(np.array(NRSS_va).std()))
    #     i = i+1









