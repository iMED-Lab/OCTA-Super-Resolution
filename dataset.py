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
        self.img_lst, self.gt_lst, self.img_lst1= self.get_dataPath(root, isTraining)
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
        imgPath1 = self.img_lst1[index]
        self.name = imgPath.split("/")[-1]

        gtPath = self.gt_lst[index]
        simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        test_simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        img = Image.open(imgPath).convert('L')
        img1 = Image.open(imgPath1)
        gt = Image.open(gtPath)
        img = img.resize((512,512))

        # gt = Image.open(gtPath).convert("L").resize(self.scale_size, Image.BICUBIC)

        if self.channel == 1:
            img = img.convert("L")
            img1 = img1.convert("L")
            gt = gt.convert('L')
        else:
            img = img.convert("RGB")
            img1 = img1.convert("RGB")
            gt = gt.convert('RGB')

        # 裁剪为（128,128）

        if self.isTraining:

            img = np.array(img)
            img1 = np.array(img1)
            gt = np.array(gt)  # 转为数组
            # img,img1, gt = self.RandomCrop(img, img1,gt, crop_factor=(64,64))  # 随机裁剪为128*128输入
            img = Image.fromarray(img)
            img1 = Image.fromarray(img1)

            gt = Image.fromarray(gt)  # array 转为imae
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img = simple_transform(img)
            img1 = simple_transform(img1)
            gt = test_simple_transform(gt)
            # img = img.resize(self.scale_size, Image.BICUBIC)
        else:
            # img = img.resize(self.scale_size, Image.BICUBIC)
            img = np.array(img)
            img1 = np.array(img1)
            gt = np.array(gt)  # 转为数组
            # img, img1, gt = self.RandomCrop(img, img1, gt, crop_factor=(152, 152))  # 随机裁剪为128*128输入
            img = Image.fromarray(img)
            img1 = Image.fromarray(img1)

            gt = Image.fromarray(gt)  # array 转为imae
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img = simple_transform(img)
            img1 = simple_transform(img1)
            gt = test_simple_transform(gt)



        return  img,img1,gt


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
        # a = random.randint(0, h - crop_factor[0])
        # b = random.randint(0, d - crop_factor[1])

        image = image[y:y + crop_factor[0], x:x + crop_factor[1]]
        # image1 = image1[a:a + crop_factor[0], b:b + crop_factor[1]]
        label = label[(y*2):((y*2 + crop_factor[0]*2)), (x*2):((x*2 + crop_factor[1]*2))]

        return image, label
    
    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            file_dir_t = os.path.join(root + "/rose/optovue6x6_152")
            file_dir_s = os.path.join(root + "/rose/optovue3x3_152")
            gt_dir_s = os.path.join(root + "/rose/optovue3x3_304")

            # file_dir_t = os.path.join(root + "/zeiss/6x6_256")
            # file_dir_s = os.path.join(root + "/zeiss/3x3_256_bis")
            # gt_dir_s = os.path.join(root + "/zeiss/3x3_512")
        else:
            # file_dir_t = os.path.join(root + "/zeiss/test_6_256")
            # file_dir_s = os.path.join(root + "/zeiss/test_3_256")
            # gt_dir_s = os.path.join(root + "/zeiss/test_3_512")
            file_dir_t = os.path.join(root + "/Normal")
            file_dir_s = os.path.join(root + "/Normal")
            gt_dir_s = os.path.join(root + "/Normal")
            # file_dir_t = os.path.join(root + "/rose_all_1/test_6x6_152_1")
            # file_dir_s = os.path.join(root + "/rose_all_1/test_3x3_152_1")
            # gt_dir_s = os.path.join(root + "/rose_all_1/test_3x3_304")
            #
        # img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        # gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        img_lst_s = []
        gt_lst_s = []
        img_lst_t = []
        file_list_s = os.listdir(file_dir_s)
        file_list_s.sort()
        for item in file_list_s:
           file_path_s = os.path.join(file_dir_s,item)
           img_lst_s.append(file_path_s)

        gt_list_s = os.listdir(gt_dir_s)
        gt_list_s.sort()
        for item in gt_list_s:
            gt_path_s = os.path.join(gt_dir_s, item)
            gt_lst_s.append(gt_path_s)
        if isTraining:
            file_list_t = os.listdir(file_dir_t)
        else:
            file_list_t = os.listdir(file_dir_t)
            file_list_t.sort()

        for item in file_list_t:
            file_path_t = os.path.join(file_dir_t, item)
            img_lst_t.append(file_path_t)
        
        return img_lst_s, gt_lst_s ,img_lst_t
    
    def getFileName(self):
        return self.name


class OCTA_zeiss(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(512, 512)):
        super(OCTA_zeiss, self).__init__()
        self.img_lst_lr,  self.img_lst_hr = self.get_dataPath(root, isTraining)
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
        imgPath_lr = self.img_lst_lr[index]
        imgPath_hr = self.img_lst_hr[index]
        self.name =  imgPath_lr.split("/")[-1]

        simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        test_simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        img_lr = Image.open(imgPath_lr)
        img_hr = Image.open(imgPath_hr)
        #
        if self.channel == 1:
            img_lr = img_lr.convert("L")
            img_hr = img_hr.convert("L")
        else:
            img_lr = img_lr.convert("RGB")
            img_hr = img_hr.convert("RGB")

        # 裁剪为（128,128）

        if self.isTraining:

            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            img_lr, img_hr, img_hr1 = self.RandomCrop(img_lr, img_hr, crop_factor=(64, 64))

            #  随机裁剪为128*128输入
            img_lr = Image.fromarray(img_lr)
            img_hr = Image.fromarray(img_hr)
            img_hr1 = Image.fromarray(img_hr1)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-45, 45)
            # img_lr = img_lr.rotate(angel)
            # angel = random.randint(-45, 45)
            # img_hr = img_hr.rotate(angel)
            # angel = random.randint(-45, 45)
            # img_hr1 = img_hr1.rotate(angel)

            img_lr = simple_transform(img_lr)
            img_hr = simple_transform(img_hr)
            img_hr1 = simple_transform(img_hr1)

            return img_lr, img_hr, img_hr1
            # img = img.resize(self.scale_size, Image.BICUBIC)
        else:
            # img_lr = img_lr.resize(self.scale_size, Image.BICUBIC)
            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            # img_lr, img_hr = self.RandomCrop(img_lr, img_hr, crop_factor=(256, 256))

            # 随机裁剪为128*128输入
            img_lr = Image.fromarray(img_lr)
            img_hr = Image.fromarray(img_hr)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img_lr = test_simple_transform(img_lr)
            img_hr = test_simple_transform(img_hr)

            return img_lr, img_hr




    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst_lr)

    # def RandomCrop(self, image, image1, crop_factor=(0, 0)):
    #     """
    #     Make a random crop of the whole volume
    #     :param image:
    #     :param label:
    #     :param crop_factor: The crop size that you want to crop
    #     :return:
    #     """
    #     h, d = image.shape
    #     # z = random.randint(0, w - crop_factor[0])
    #     # y = random.randint(0, h - crop_factor[0])
    #     # x = random.randint(0, d - crop_factor[1])
    #     # a = random.randint(0, h - crop_factor[0])
    #     # b = random.randint(0, d - crop_factor[1])
    #
    #     image = image[int(h//4):int(h//4) + crop_factor[0], int(d//4):(d//4) + crop_factor[1]]
    #     image1 = image1
    #     return image, image1
    def RandomCrop(self, image, label, crop_factor=(0, 0)):
        """
        Make a random crop of the whole volume
        :param image:
        :param label:
        :param crop_factor: The crop size that you want to crop
        :return:
        """
        h0, d0, _ = image.shape
        h = h0//2
        d = d0//2
        label0 = image[106:406,106:406,:]
        h1, d1, _ = label0.shape
        # print(h1)
        h2 = h1//2
        d2 = d1//2

        x = random.randint(0, (h // 2 - crop_factor[0])) or random.randint(h * 3 // 2, (h * 2 - crop_factor[0]))
        y = random.randint(0, d // 2 - crop_factor[1]) or random.randint(d * 3 // 2, (d * 2 - crop_factor[1]))

        # z = random.randint(0, w - crop_factor[0])

        a = random.randint(0, (h2 // 2 - crop_factor[0])) or random.randint(h2 * 3 // 2, (h2 * 2 - crop_factor[0]))
        b = random.randint(0, d2 // 2 - crop_factor[1]) or random.randint(d2 * 3 // 2, (d2 * 2 - crop_factor[1]))
        c = random.randint(0, h1 - crop_factor[0])
        d = random.randint(0, d1 - crop_factor[1])


        label1 = label0[a:a + crop_factor[0], b:b + crop_factor[1],:]
        label2 = label0[c:((c + crop_factor[0])), (d):((d + crop_factor[1])), :]
        # image1 = image1[a:a + crop_factor[0], b:b + crop_factor[1]]
        image1 = image[y:((y + crop_factor[0])), (x):((x + crop_factor[1])),:]


        return image1, label1, label2

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            file_dir_lr = os.path.join(root, "6x6_ori")
            file_dir_hr = os.path.join(root, "6x6_256")
        else:
            file_dir_lr = os.path.join(root, 'wen')
            file_dir_hr = os.path.join(root, "wen")

        img_lst_lr = []
        img_lst_hr = []
        file_list_lr = os.listdir(file_dir_lr)
        file_list_lr.sort()
        for item in file_list_lr:
            file_path_s = os.path.join(file_dir_lr, item)
            img_lst_lr.append(file_path_s)

        file_list_hr = os.listdir(file_dir_hr)
        file_list_hr.sort()

        for item in file_list_hr:
            file_path_hr = os.path.join(file_dir_hr, item)
            img_lst_hr.append(file_path_hr)

        return img_lst_lr, img_lst_hr

    def getFileName(self):
        return self.name


class OCTA_zeiss_de(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(512, 512)):
        super(OCTA_zeiss_de, self).__init__()
        self.img_lst_lr,  self.img_lst_hr = self.get_dataPath(root, isTraining)
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
        imgPath_lr = self.img_lst_lr[index]
        imgPath_hr = self.img_lst_hr[index]
        self.name =  imgPath_lr.split("/")[-1]

        simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        test_simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        img_lr = Image.open(imgPath_lr)
        img_hr = Image.open(imgPath_hr)
        #
        if self.channel == 1:
            img_lr = img_lr.convert("L")
            img_hr = img_hr.convert("L")
        else:
            img_lr = img_lr.convert("RGB")
            img_hr = img_hr.convert("RGB")

        # 裁剪为（128,128）

        if self.isTraining:

            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            img_lr, img_hr= self.RandomCrop(img_lr, img_hr, crop_factor=(96, 96))

            #  随机裁剪为128*128输入
            img_lr = Image.fromarray(img_lr)
            img_hr = Image.fromarray(img_hr)
            # img_hr1 = Image.fromarray(img_hr1)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-45, 45)
            # img_lr = img_lr.rotate(angel)
            # angel = random.randint(-45, 45)
            # img_hr = img_hr.rotate(angel)
            # angel = random.randint(-45, 45)
            # img_hr1 = img_hr1.rotate(angel)

            img_lr = simple_transform(img_lr)
            img_hr = simple_transform(img_hr)
            # img_hr1 = simple_transform(img_hr1)

            return img_lr, img_hr
            # img = img.resize(self.scale_size, Image.BICUBIC)
        else:
            # img_lr = img_lr.resize(self.scale_size, Image.BICUBIC)
            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            # img_lr, img_hr = self.RandomCrop(img_lr, img_hr, crop_factor=(256, 256))

            # 随机裁剪为128*128输入
            img_lr = Image.fromarray(img_lr)
            img_hr = Image.fromarray(img_hr)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img_lr = test_simple_transform(img_lr)
            img_hr = test_simple_transform(img_hr)

            return img_lr, img_hr




    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst_lr)

    # def RandomCrop(self, image, image1, crop_factor=(0, 0)):
    #     """
    #     Make a random crop of the whole volume
    #     :param image:
    #     :param label:
    #     :param crop_factor: The crop size that you want to crop
    #     :return:
    #     """
    #     h, d = image.shape
    #     # z = random.randint(0, w - crop_factor[0])
    #     # y = random.randint(0, h - crop_factor[0])
    #     # x = random.randint(0, d - crop_factor[1])
    #     # a = random.randint(0, h - crop_factor[0])
    #     # b = random.randint(0, d - crop_factor[1])
    #
    #     image = image[int(h//4):int(h//4) + crop_factor[0], int(d//4):(d//4) + crop_factor[1]]
    #     image1 = image1
    #     return image, image1
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
        a = random.randint(0, h - crop_factor[0])
        b = random.randint(0, d - crop_factor[1])

        image1 = image[y:y + crop_factor[0], x:x + crop_factor[1]]
        # image1 = image1[a:a + crop_factor[0], b:b + crop_factor[1]]
        label1 = label[(a * 2):((a * 2 + crop_factor[0] * 2)), (b * 2):((b * 2 + crop_factor[1] * 2))]

        return image1, label1

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            file_dir_lr = os.path.join(root, "6x6_256")
            file_dir_hr = os.path.join(root, "3x3_512")
        else:
            file_dir_lr = os.path.join(root, "test_6_256")
            file_dir_hr = os.path.join(root, "test_3_512")

        img_lst_lr = []
        img_lst_hr = []
        file_list_lr = os.listdir(file_dir_lr)
        file_list_lr.sort()
        for item in file_list_lr:
            file_path_s = os.path.join(file_dir_lr, item)
            img_lst_lr.append(file_path_s)

        file_list_hr = os.listdir(file_dir_hr)
        file_list_hr.sort()

        for item in file_list_hr:
            file_path_hr = os.path.join(file_dir_hr, item)
            img_lst_hr.append(file_path_hr)

        return img_lst_lr, img_lst_hr

    def getFileName(self):
        return self.name

class OCTA_rose_de(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(512, 512)):
        super(OCTA_rose_de, self).__init__()
        self.img_lst_lr,  self.img_lst_hr = self.get_dataPath(root, isTraining)
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
        imgPath_lr = self.img_lst_lr[index]
        imgPath_hr = self.img_lst_hr[index]
        self.name =  imgPath_lr.split("/")[-1]

        simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        test_simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        img_lr = Image.open(imgPath_lr)
        img_hr = Image.open(imgPath_hr)
        #
        if self.channel == 1:
            img_lr = img_lr.convert("L")
            img_hr = img_hr.convert("L")
        else:
            img_lr = img_lr.convert("RGB")
            img_hr = img_hr.convert("RGB")

        # 裁剪为（128,128）

        if self.isTraining:

            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            img_lr, img_hr= self.RandomCrop(img_lr, img_hr, crop_factor=(96, 96))

            #  随机裁剪为128*128输入
            img_lr = Image.fromarray(img_lr)
            img_hr = Image.fromarray(img_hr)
            # img_hr1 = Image.fromarray(img_hr1)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-45, 45)
            # img_lr = img_lr.rotate(angel)
            # angel = random.randint(-45, 45)
            # img_hr = img_hr.rotate(angel)
            # angel = random.randint(-45, 45)
            # img_hr1 = img_hr1.rotate(angel)

            img_lr = simple_transform(img_lr)
            img_hr = simple_transform(img_hr)
            # img_hr1 = simple_transform(img_hr1)

            return img_lr, img_hr
            # img = img.resize(self.scale_size, Image.BICUBIC)
        else:
            # img_lr = img_lr.resize(self.scale_size, Image.BICUBIC)
            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            # img_lr, img_hr = self.RandomCrop(img_lr, img_hr, crop_factor=(256, 256))

            # 随机裁剪为128*128输入
            img_lr = Image.fromarray(img_lr)
            img_hr = Image.fromarray(img_hr)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img_lr = test_simple_transform(img_lr)
            img_hr = test_simple_transform(img_hr)

            return img_lr, img_hr




    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst_lr)

    # def RandomCrop(self, image, image1, crop_factor=(0, 0)):
    #     """
    #     Make a random crop of the whole volume
    #     :param image:
    #     :param label:
    #     :param crop_factor: The crop size that you want to crop
    #     :return:
    #     """
    #     h, d = image.shape
    #     # z = random.randint(0, w - crop_factor[0])
    #     # y = random.randint(0, h - crop_factor[0])
    #     # x = random.randint(0, d - crop_factor[1])
    #     # a = random.randint(0, h - crop_factor[0])
    #     # b = random.randint(0, d - crop_factor[1])
    #
    #     image = image[int(h//4):int(h//4) + crop_factor[0], int(d//4):(d//4) + crop_factor[1]]
    #     image1 = image1
    #     return image, image1
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
        a = random.randint(0, h - crop_factor[0])
        b = random.randint(0, d - crop_factor[1])

        image1 = image[y:y + crop_factor[0], x:x + crop_factor[1]]
        # image1 = image1[a:a + crop_factor[0], b:b + crop_factor[1]]
        label1 = label[(a * 2):((a * 2 + crop_factor[0] * 2)), (b * 2):((b * 2 + crop_factor[1] * 2))]

        return image1, label1

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            file_dir_lr = os.path.join(root, "optovue6x6_152")
            file_dir_hr = os.path.join(root, "optovue3x3_304")
        else:
            file_dir_lr = os.path.join(root, "test_6x6-152")
            file_dir_hr = os.path.join(root, "test_3x3-304")

        img_lst_lr = []
        img_lst_hr = []
        file_list_lr = os.listdir(file_dir_lr)
        file_list_lr.sort()
        for item in file_list_lr:
            file_path_s = os.path.join(file_dir_lr, item)
            img_lst_lr.append(file_path_s)

        file_list_hr = os.listdir(file_dir_hr)
        file_list_hr.sort()

        for item in file_list_hr:
            file_path_hr = os.path.join(file_dir_hr, item)
            img_lst_hr.append(file_path_hr)

        return img_lst_lr, img_lst_hr

    def getFileName(self):
        return self.name

class OCTA_zeiss_origin(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(256, 256)):
        super(OCTA_zeiss_origin, self).__init__()
        self.img_lst_lr,  self.img_lst_hr, self.img_lst_lr_ori = self.get_dataPath(root, isTraining)
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
        imgPath_lr = self.img_lst_lr[index]
        imgPath_hr = self.img_lst_hr[index]
        imgPath_lr_ori = self.img_lst_lr_ori[index]
        self.name =  imgPath_lr_ori.split("/")[-1]

        simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        test_simple_transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.ToTensor()])

        img_lr = Image.open(imgPath_lr)
        img_hr = Image.open(imgPath_hr)
        img_lr_ori = Image.open(imgPath_lr_ori)

        if self.channel == 1:
            img_lr = img_lr.convert("L")
            img_hr = img_hr.convert("L")
            img_lr_ori = img_lr_ori.convert("L")
        else:
            img_lr = img_lr.convert("RGB")
            img_hr = img_hr.convert("RGB")
            img_lr_ori = img_lr_ori.convert("RGB")

        # 裁剪为（128,128）

        if self.isTraining:

            img_lr = np.array(img_lr)
            img_hr = np.array(img_hr)
            img_lr_ori = np.array(img_lr_ori)
            img_lr, img_hr, img_lr_ori = self.RandomCrop(img_lr, img_hr, img_lr_ori, crop_factor=(128, 128))

            #  随机裁剪为128*128输入
            img_lr = Image.fromarray(img_lr)
            img_hr = Image.fromarray(img_hr)
            img_lr_ori = Image.fromarray(img_lr_ori)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img_lr = simple_transform(img_lr)
            img_hr = simple_transform(img_hr)
            img_lr_ori = simple_transform(img_lr_ori)

            return img_lr, img_hr,img_lr_ori
            # img = img.resize(self.scale_size, Image.BICUBIC)
        else:
            # img = img.resize(self.scale_size, Image.BICUBIC)
            img_lr_ori = np.array(img_lr_ori)
            # img_lr, img_hr = self.RandomCrop(img_lr, img_hr, crop_factor=(256, 256))

            # 随机裁剪为128*128输入
            img_lr_ori = Image.fromarray(img_lr_ori)
            # img_hr = img_hr.resize(self.scale_size, Image.BICUBIC)
            # rotate = 10
            # angel = random.randint(-rotate, rotate)
            # img = img.rotate(angel)
            # gt = gt.rotate(angel)
            img_lr_ori = test_simple_transform(img_lr_ori)

            return  img_lr_ori



    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst_lr)

    # def RandomCrop(self, image, image1, crop_factor=(0, 0)):
    #     """
    #     Make a random crop of the whole volume
    #     :param image:
    #     :param label:
    #     :param crop_factor: The crop size that you want to crop
    #     :return:
    #     """
    #     h, d = image.shape
    #     # z = random.randint(0, w - crop_factor[0])
    #     # y = random.randint(0, h - crop_factor[0])
    #     # x = random.randint(0, d - crop_factor[1])
    #     # a = random.randint(0, h - crop_factor[0])
    #     # b = random.randint(0, d - crop_factor[1])
    #
    #     image = image[int(h//4):int(h//4) + crop_factor[0], int(d//4):(d//4) + crop_factor[1]]
    #     image1 = image1
    #     return image, image1
    def RandomCrop(self, image, label, image_ori,crop_factor=(0, 0)):
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
        a = random.randint(0, h*2 - crop_factor[0])
        b = random.randint(0, d*2 - crop_factor[1])


        image = image[y:y + crop_factor[0], x:x + crop_factor[1]]
        # image1 = image1[a:a + crop_factor[0], b:b + crop_factor[1]]
        label = label[y*2:((y*2 + crop_factor[0]*2)), (x*2):((x*2 + crop_factor[1]*2))]
        image_ori = image_ori[a:a + crop_factor[0], b:b + crop_factor[1]]
        return image, label, image_ori

    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            file_dir_lr = os.path.join(root, "6x6_256")
            file_dir_hr = os.path.join(root, "3x3_512")
            file_dir_lr_ori = os.path.join(root, "6x6_ori")
            img_lst_lr = []
            img_lst_hr = []
            img_lst_lr_ori = []
            file_list_lr = os.listdir(file_dir_lr)
            file_list_lr.sort()
            for item in file_list_lr:
                file_path_s = os.path.join(file_dir_lr, item)
                img_lst_lr.append(file_path_s)

            file_list_hr = os.listdir(file_dir_hr)
            file_list_hr.sort()

            for item in file_list_hr:
                file_path_hr = os.path.join(file_dir_hr, item)
                img_lst_hr.append(file_path_hr)

            file_list_lr_ori = os.listdir(file_dir_lr_ori)
            file_list_lr_ori .sort()
            for item in file_list_lr_ori :
                file_path_s_ori  = os.path.join(file_dir_lr_ori, item)
                img_lst_lr_ori.append(file_path_s_ori)

            return img_lst_lr, img_lst_hr, img_lst_lr_ori

        else:
            file_dir_lr_ori = os.path.join(root, "test_6x6_ori")
            img_lst_lr_ori = []
            file_list_lr_ori = os.listdir(file_dir_lr_ori)
            file_list_lr_ori.sort()
            for item in file_list_lr_ori:
                file_path_s_ori = os.path.join(file_dir_lr_ori, item)
                img_lst_lr_ori.append(file_path_s_ori)

            return img_lst_lr_ori, img_lst_lr_ori, img_lst_lr_ori


    def getFileName(self):
        return self.name

