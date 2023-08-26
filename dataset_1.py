# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
# import cv2
import random
from PIL import Image
import numpy as np

import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch


# 随机裁剪，保证image和label的裁剪方式一致
def random_crop(image, label, crop_size=(256, 256)):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)

    return image, label


class CT_norm(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(304, 304)):
        super(CT_norm, self).__init__()
        self.img_lst, self.gt_lst= self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""

        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]

        self.name = imgPath.split("/")[-1]

        gtPath = self.gt_lst[index]
        simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        test_simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        img = Image.open(imgPath)

        gt = Image.open(gtPath)
        # gt = gt.resize((512, 512), Image.BICUBIC)

        # gt = Image.open(gtPath).convert("L").resize(self.scale_size, Image.BICUBIC)

        if self.channel == 1:
            img = img.convert("L")
            gt = gt.convert('L')
        else:
            img = img.convert("RGB")
            gt = gt.convert('RGB')

        # 裁剪为（128,128）

        if self.isTraining:

            img = np.array(img)

            gt = np.array(gt)  # 转为数组
            img, gt = self.RandomCrop(img, gt, crop_factor=(128, 128))  # 随机裁剪为128*128输入
            img = Image.fromarray(img)

            gt = Image.fromarray(gt)  # array 转为imae
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img = simple_transform(img)

            gt = test_simple_transform(gt)
            # img = img.resize(self.scale_size, Image.BICUBIC)
        else:
            # img = img.resize(self.scale_size, Image.BICUBIC)
            img = np.array(img)

            gt = np.array(gt)  # 转为数组
            #  随机裁剪为128*128输入
            img = Image.fromarray(img)
            gt = Image.fromarray(gt)  # array 转为imae
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img = simple_transform(img)
            gt = simple_transform(gt)

        return img, gt

    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)

    def RandomCrop(self, image, label, crop_factor=(0, 0)):
        """
        Make a random crop of the whole volume
        :param image:
        :param label:
        :param crop_factor: The crop size that you want to crop
        :return:
        """
        h, d = image.shape
        # z = random.randint(0, w - crop_factor[0])
        y = random.randint(0, h - crop_factor[0])
        x = random.randint(0, d - crop_factor[1])

        image = image[y:y + crop_factor[0], x:x + crop_factor[1]]

        label = label[(y * 2):((y * 2 + crop_factor[0] * 2)), (x * 2):((x * 2 + crop_factor[1] * 2))]
        return image,  label

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            file_dir_s = os.path.join(root + "/rose/optovue3x3_152_bi")
            gt_dir_s = os.path.join(root + "/rose/optovue3x3_304")
            # file_dir_s = os.path.join(root + "/zeiss/3x3_256_bis")
            # gt_dir_s = os.path.join(root + "/zeiss/3x3_512")
        else:
            file_dir_s = os.path.join(root + "/rose_all/test_3x3_152_1")
            gt_dir_s = os.path.join(root + "/rose_all/test_3x3_304")
            # file_dir_s = os.path.join(root + "/zeiss/test_3_256")
            # gt_dir_s = os.path.join(root + "/zeiss/test_3_512")

        # img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        # gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        img_lst_s = []
        gt_lst_s = []
        img_lst_t = []
        file_list_s = os.listdir(file_dir_s)
        file_list_s.sort()
        for item in file_list_s:
            file_path_s = os.path.join(file_dir_s, item)
            img_lst_s.append(file_path_s)

        gt_list_s = os.listdir(gt_dir_s)
        gt_list_s.sort()
        for item in gt_list_s:
            gt_path_s = os.path.join(gt_dir_s, item)
            gt_lst_s.append(gt_path_s)


        return img_lst_s, gt_lst_s

    def getFileName(self):
        return self.name
