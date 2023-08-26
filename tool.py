#coding=utf-8
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from visualize import Visualizer
import os
import torch.nn.functional as F
import SimpleITK as sitk
#
# vis = Visualizer(env='fft')
# image_path = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/dataset/train/zeiss/6x6_256/xformed_DR052OD6x6_to_DR052OD3x3.png'
# image  = Image.open(image_path).convert('L')
# image = transforms.ToTensor()(image)
# fft = torch.rfft(image, 3, onesided=True)
# print(fft.size())
# fft_mag = torch.log(torch.abs(fft))
# print(fft_mag.size())
# # fft_mag = torch.nn.Sigmoid()(fft_mag)
# vis.img(name = 'gen', img_ = fft_mag[0:1,:,:])
file_path = '/media/imed/My Passport/123/select_thrombus'
file_path0 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/左心耳分割/thrombus/img'
file_path00 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/左心耳分割/thrombus/mask'
file_list = os.listdir(file_path)
file_list.sort()
i = 0
for item in file_list:
    file_path1 = os.path.join(file_path,item)
    file_list1 = os.listdir(file_path1)
    for item1 in file_list1:
        file_path2 = os.path.join(file_path1, item1)
        file_list2 = os.listdir(file_path2)
        for item2 in file_list2:
            if item2.endswith('nii.gz'):
                image_path = os.path.join(file_path2,item2[:-7]+'.png')
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    file_path_final = os.path.join(file_path0, item, item1)
                    if not os.path.exists(file_path_final):
                        os.makedirs(file_path_final)
                    image.save(os.path.join(file_path_final, item2[:-7] + '.png'))
                    image_path1 = os.path.join(file_path2, item2)
                    img = sitk.ReadImage(image_path1)
                    img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                    img = sitk.GetArrayFromImage(img)
                    img = Image.fromarray(img[0,:,:])
                    file_path_final1 = os.path.join(file_path00, item, item1)
                    if not os.path.exists(file_path_final1):
                        os.makedirs(file_path_final1)
                    img.save(os.path.join(file_path_final1, item2[:-7] + '.png'))
                    i+=1
                    # img = sitk.GetArrayFromImage(img)
                    # print(img.shape)

                    # out = sitk.GetImageFromArray(img)
                    #
                    # sitk.WriteImage(out, 'xxxxx.nii.gz')
                    #
                    print(image_path)
print(i)
            # if '，' in item :
            #     item_ = item.replace('，','_')
            # else:
            #     item_ = item
            # if item1[:2] == '选，':
            #     item1_ = 'select_' + item1[2:]
            # else:
            #     item1_ = item1
            # if item2[:2]=='选，':
            #     item2_ = 'select_' + item2[2:]
            # else:
            #     item2_ = item2

            # file_list3 = os.listdir(file_path3)
            # for item3 in file_list3:
            #     image_path = os.path.join(file_path3,item3)



# f = open("data.txt","r")
# data = f.readlines()
#
# print(data)
# ref = 'ABCDEFGHIJKLMNO'
# data = []
# for line in open("data.txt","r"):
#     data.append(line[:-1])
#     new_data = []
#     new_str = data[0]
#     for i in range(len(data[0])):
#         for j in range(len(ref)):
#             if new_str[i]!=ref[j]:
#               new_str1 = new_str.replace(new_str[i],ref[j])
#               print(new_str1)

