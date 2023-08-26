# CelebA image generation using DCGAN
import torch
from torch.autograd import Variable
from resnet import D_net
from torch.utils.data import DataLoader
from dataset import OCTA_zeiss
from visualize import Visualizer
import os
import torch.nn as nn
# from models.RDN import rdn
import pytorch_ssim
from math import log10
import numpy as np
from dataset_blindsr import DatasetBlindSR
import cv2
from resnet import WaveletTransform
from voxel_main import VoxelMorph
# from networks import G_Module
from models.EDSR import edsr
from torchvision import transforms
import time
from models.deg_arch import DegModel
import lpips
from resnet import WaveletTransform
from models.new_model import rdn
from mmd_loss import MMD_loss
from torch.nn import functional as F
from resnet import Vgg
# from Trans_U_NET.networks import vit_seg_configs,vit_seg_modeling
# config = vit_seg_configs.get_r50_b16_config()
import random

# Parameters
image_size = 64
G_input_dim = 100
G_output_dim = 3
D_input_dim = 3
D_output_dim = 1
num_filters = [1024, 512, 256, 128]

learning_rate_G = 0.00005
learning_rate_G1 = 0.0001
learning_rate_D = 0.0001
betas = (0.5, 0.999)
train_batch_size = 8
test_batch_size = 1
num_epochs = 400
file_dir = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/model'
# CT dataset
file_path = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/dataset/train/zeiss/3x3_512'
file_path1 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/dataset/train/zeiss/test_6_256'
file_path2 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/dataset/train/zeiss/6x6_256'
train_data_loader = DataLoader(dataset = DatasetBlindSR(root = file_path, root_1=file_path2, isTraining = True),
                                          batch_size = train_batch_size,
                                          shuffle = True)
test_data_loader = DataLoader(dataset = DatasetBlindSR(root = file_path1, root_1=None, isTraining = False),
                                          batch_size = test_batch_size,
                                          shuffle = False)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def gaussian(image,mean,std):
    b,c,h,w = image.shape
    noise = Variable((torch.randn([b,c,h,w])*std+mean).cuda())
    return noise

def EMA(model1, model2, m):
    with torch.no_grad():
        for param_q, param_k in zip(model1.parameters(), model2.parameters()):
            param_k.data.mul_(m).add_((1-m)*param_q.detach().data)
    return model2

def weighted_mse_loss(input, target, weight):
    input = input.view(input.shape[0],-1)
    target = target.view(target.shape[0], -1)

    return (weight * (torch.mean(((input - target) ** 2),dim=1))).sum()

def Normalize(tensor):
    tensor = tensor.view(-1)
    out = tensor/(tensor.sum()+1e-12)
    return out


# Plot losses
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    # device = torch.device('cuda', 6)

    G_model_path = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/model/haohy_domain_ada_sr_zeiss+sr_domain/G1-317.pth'
    G = torch.load(G_model_path).cuda()
    G = G.module
    G1 = rdn().cuda()
    model_dict1 = G.state_dict()
    model_dict2 = G1.state_dict()
    model_list1 = list(model_dict1.keys())
    model_list2 = list(model_dict2.keys())
    # len1 = len(model_list1)
    # len2 = len(model_list2)
    # print(len1,len2)
    # m, n = 0, 0
    # layers = []
    # while True:
    #
    #     if m >= len1 or n >= len2:
    #         break
    #     layername1, layername2 = model_list1[m], model_list2[n]
    #     w1, w2 = model_dict1[layername1], model_dict2[layername2]
    #     if w1.shape != w2.shape:
    #         m += 1
    #         continue
    #     model_dict2[layername2] = model_dict1[layername1]
    #     # print(n)
    #     layers.append(n)
    #     m += 1
    #     n += 1
    G1.load_state_dict(model_dict1)
    # for i, layer in enumerate(G1.layers[0].layers):
    #     if i in layers:
    #         print(11)
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True
    # for i, param in enumerate(G1.parameters()):
    #     if i in layers:
    #         print(i)
    #         param.requires_grad = False
    # G_model_path0 = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/model/haohy_domain_ada_sr_rose_rdn_6x6/G-77.pth'
    # G0 = torch.load(G_model_path0).cuda()
    # G0 = G0.module
    # for param in G.parameters():
    #     param.requires_grad = False
    # G1 = rdn().cuda()
    # H_L = H_L().cuda()
    # G1 = rdn()
    D = D_net().cuda()
    # vm = VoxelMorph((1, 128, 128), is_2d=True)
    # P = P_net()
    # E = Vgg().cuda()
    # D1 = D1_net()
    G = nn.DataParallel(G).cuda()
    D = nn.DataParallel(D).cuda()
    # D1 = nn.DataParallel(D1).cuda()
    G1 = nn.DataParallel(G1).cuda()
    # loss_vgg = lpips.LPIPS(net='vgg').cuda()
    # D1 = nn.DataParallel(D1).cuda()
    # Loss function
    criterion1 = torch.nn.L1Loss().cuda()
    criterion2 = torch.nn.BCELoss(reduce=True).cuda()
    criterion3 = torch.nn.BCELoss(reduce=True).cuda()
    criterion4 = torch.nn.MSELoss(reduce=True).cuda()
    criterion5 = torch.nn.MSELoss(reduce=True).cuda()
    ssim_loss = pytorch_ssim.SSIM(size_average=False).cuda()
    ssim_loss_ = pytorch_ssim.SSIM(size_average=True).cuda()
    mmd_loss  = MMD_loss().cuda()
    # wavelet_dec = WaveletTransform(dec=True).cuda()
    # Optimizers
    # G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate_G, betas=betas)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate_D, betas=betas)
    # D1_optimizer = torch.optim.Adam(D1.parameters(), lr=learning_rate, betas=betas)
    G1_optimizer = torch.optim.Adam(G1.parameters(), lr=learning_rate_G1, betas=betas)
    # Training GAN
    D_avg_losses = []
    G_avg_losses = []
    # Fixed noise for test
    # num_test_samples = 5*5
    # fixed_noise = torch.randn(num_test_samples, G_input_dim).view(-1, G_input_dim, 1, 1)
    file_name = 'haohy_domain_ada_sr_zeiss+sr_domain_ada_new_model3'
    vis = Visualizer(env=file_name)
    j = 0
    for epoch in range(num_epochs):
        # if epoch % 40 == 0:
        #     j += 1
        # G.train()
        G1.train()
        D.train()
        # loss_vgg.train()
        # minibatch training
        D_mean_loss = 0.0
        D1_mean_loss = 0.0
        G1_mean_loss = 0.0
        G2_mean_loss = 0.0
        G3_mean_loss = 0.0
        ssim_mean_loss = 0.0
        ssim_mean_loss2 = 0.0
        lpips_mean_loss = 0.0
        # E_mean_loss = 0.0
        # for k in range(2):
        # H_L_gen = G_Module(norm='Instance').cuda()
        for i, (images, images1, images2) in enumerate(train_data_loader):
            # model_id = random.randint(0,23)
            # model_root = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/super_resolution_dataset/model/haohy_domain_ada_sr_zeiss+PDM'
            # model_dir = os.listdir(model_root)
            # model = torch.load(os.path.join(model_root,model_dir[model_id]))
            # for param in model.parameters():
            #     param.requires_grad = False
            # image data
            x_lr = images.cuda()
            x_hr = images1.cuda()
            x_lr_r = images2.cuda()
            image_size = x_lr.size()
            # Train discriminator with real data
            # D_loss
            x_hr_gen0 = G(x_lr)
            x_hr_gen = G1(x_lr)
            x_hr_gen_r = G1(x_lr_r)
            # ori_wavelet = wavelet_dec(x_hr_gen)
            # ori_wavelet_lf = ori_wavelet[:, 2:3, :, :]
            # gen_image_wavelet = wavelet_dec(x_hr_gen_r)
            # gen_image_wavelet_lf = gen_image_wavelet[:, 2:3, :, :]
            ssim_score = ssim_loss(x_hr_gen0, x_hr[:, 0:1, :, :])
            # ssim_score1 = ssim_score.detach()

            ssim_loss1_weight = (F.relu(ssim_score - 0.6)) * ((1) / (ssim_score - 0.6 + 1e-12))
            print(ssim_loss1_weight)
            mmd_score = ((1 - Normalize(mmd_loss(x_lr,x_lr_r))))
            mmd_score1 = Normalize((ssim_loss1_weight))
            print(mmd_score)
            # ssim_1 = 1 - ssim_loss_(x_lr[:,0:1,:,:],x_lr_r[:,0:1,:,:])
            # print(mmd_score)
            D_real_decision = D(x_hr_gen).squeeze()
            # print(D_real_decision.shape)
            mini_batch = D_real_decision.size()
            y_real_ = Variable(torch.ones(mini_batch).cuda())
            y_fake_ = Variable(torch.zeros(mini_batch).cuda())
            D_real_loss = criterion5(D_real_decision, y_real_)
            D_fake_decision = D(x_hr_gen_r).squeeze()
            D_fake_loss = criterion5(D_fake_decision, y_fake_)
            # D1_loss
            # D1_fake_decision = D(gen_image1).squeeze()
            # D1_fake_loss = criterion4(D1_fake_decision, y_fake_)
            D_loss = D_real_loss + D_fake_loss
            # Back propagation
            D.zero_grad()
            D_loss.backward(retain_graph=True)
            # ssim_loss1.backward(retain_graph=True)
            # D1_loss.backward()
            D_optimizer.step()
            D_mean_loss += D_loss.item()
            vis.plot(name='D_loss', y=D_mean_loss / (i + 1))
            # G_loss
            # x_hr_gen = G1(x_lr)
            # x_hr_gen0 = G(x_lr)
            # x_hr_gen = G1(x_lr)
            # x_hr_gen_r = G1(x_lr_r)
            # ori_wavelet = wavelet_dec(x_hr_gen)
            # ori_wavelet_lf = ori_wavelet[:, 2:3, :, :]
            # gen_image_wavelet = wavelet_dec(x_hr_gen_r)
            # gen_image_wavelet_lf = gen_image_wavelet[:, 2:3, :, :]
            # ssim_score = ssim_loss(x_hr_gen0, x_hr[:, 0:1, :, :])
            ssim_loss1 = 1 - (ssim_loss(x_hr_gen, x_hr[:, 0:1, :, :])).mean()
            # ssim_loss1_weight = (F.relu(ssim_score - 0.6)) * ((1) / (ssim_score - 0.6 + 1e-12))
            # mmd_score = ((1 - Normalize(mmd_loss(x_lr, x_lr_r))) * ssim_loss1_weight)
            # mmd_score = Normalize((mmd_score))
            D_fake_decision = D(x_hr_gen_r).squeeze()
            D_loss2 = criterion5(D_fake_decision, y_real_)
            mse_loss2 = criterion5(x_hr_gen_r, x_hr_gen)
            ssim_loss2 = 1 - ssim_loss_(x_hr_gen_r, x_hr[:,0:1,:,:])
            mse_loss = criterion5(x_hr_gen, x_hr[:,0:1,:,:])
            hr_filter = nn.AvgPool2d(kernel_size=8, stride=8)
            lr_filter = nn.AvgPool2d(kernel_size=4, stride=4)
            G_loss1 = 1 - ssim_loss_(lr_filter(x_lr_r[:,0:1,:,:]), hr_filter(x_hr_gen_r))


            G_loss = 4 * ssim_loss1 + 8 * mse_loss + D_loss2 + 0.01 * ssim_loss2 # Back propagatio

            D.zero_grad()
            # G.zero_grad()
            G1.zero_grad()
            # G1.zero_grad()
            G_loss.backward()
            # G_optimizer.step()
            G1_optimizer.step()

            # G1 = EMA(G, G1, 0.9)
            G1_mean_loss += G_loss.item()
            ssim_mean_loss += ssim_loss1.item()
            # lpips_mean_loss += E_loss.item()
            # gen_image_up = torch.nn.Upsample(scale_factor=2)(x_hr_256)
            x_lr_up = torch.nn.Upsample(scale_factor=2)(x_lr)
            x_lr_up_r = torch.nn.Upsample(scale_factor=2)(x_lr_r)
            out_vis = torch.cat(
                [x_hr[0:4, 0:1, :, :],x_hr_gen[0:4, :, :, :],x_lr_up[0:4,0:1,:,:], x_lr_up_r[0:4,0:1,:,:],x_hr_gen_r[0:4,:,:,:]], 0)
            vis.img(name = 'gen', img_ = out_vis, nrow=4)
            vis.plot(name = 'G1_loss', y = G1_mean_loss / (i + 1))
            # vis.plot(name='G2_loss', y=G2_mean_loss / (i + 1))
            vis.plot(name = 'ssim_loss1', y = ssim_mean_loss / (i + 1))
            # vis.plot(name='lpips_loss', y=lpips_mean_loss / (i + 1))
            # loss values
            print('Epoch [%d/%d], Step [%d/%d],  G_loss: %.4f' % (
                epoch + 1, num_epochs, i + 1, len(train_data_loader), G_loss.item()))
        checkpoint_path = os.path.join(file_dir, file_name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}.pth')
        if epoch % 1 == 0:
            torch.save(G1, checkpoint_path.format(net='G1',epoch=epoch))
        avg_psnr = 0.0
        ssim_score_total = 0.0
        ssim_score_total1 = 0.0
        with torch.no_grad():

            G1 = G1.eval()
            for i, (images) in enumerate(test_data_loader):
                # image data
                x_ = images.cuda()
                # gt_ = Variable(images1.cuda())
                image_size = x_.size()
                # x_ = x_ / 2 + 0.5

                # Train discriminator with real data
                gen_image = G1(x_)

                # x_lr_gen = gen_image.clamp(0, 1)
                # gen_image1 = torch.nn.MaxPool2d(2)(gen_image1)
                # gen_image1 = G1(gen_image1)
                # mse = criterion5(gen_image, gt_)
                # psnr = 10 * log10(1 / (mse.item()))
                # avg_psnr += psnr
                # ssim, ssim1 = ssim_loss(gen_image, gt_)
                # # ssim1 = ssim_loss_(gen_image1, gt_)
                # ssim_score_total += ssim.item()
                # ssim_score_total1 += ssim1.item()
                x1_ = torch.nn.Upsample(scale_factor=2)(x_)

                out_vis = torch.cat([x1_[0:1, 0:1, :, :], gen_image[0:1, :, :, :]], 0)
                vis.img(name='gen', img_=out_vis, nrow=2)
                # gen_image_ = gen_image.squeeze().cpu().numpy()
                gen_image1_ = gen_image.squeeze().cpu().numpy()
                name = test_data_loader.dataset.getFileName()
                # print(name, psnr, ssim.item())
                pred_img = np.array(gen_image1_ * 255, np.uint8)
                if epoch % 5 == 0:
                    path = '/media/imed/9bb6637f-ee14-4ce3-a368-215ff60d1391/py_hhy/domain_adaptation_1'
                    checkpoint_path = os.path.join(path, file_name,'{net}-{epoch}.pth'.format(net='G1',epoch=epoch))
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)

                    cv2.imwrite(os.path.join(checkpoint_path, name), pred_img)







