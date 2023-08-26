import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
from  utils import utils_blindsr as blindsr


class DatasetBlindSR(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, root, root_1, root_2, isTraining=True):
        super(DatasetBlindSR, self).__init__()
        self.training = isTraining
        self.n_channels = 3
        self.sf = 2
        self.shuffle_prob = 0.1
        self.use_sharp = False
        # self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = 128
        self.patch_size = 256
        self.real_root = root_2
        self.real_path_256 = util.get_image_paths(root_1)
        if self.real_root is not None:
            self.real_path_512 = util.get_image_paths(root_2)
        else:
            self.real_path_512 = root_1
        self.path = util.get_image_paths(root)
        self.name = ''
        # self.path_L_test = util.get_image_paths(test_root)
        # print(len(self.paths_H))

#        for n, v in enumerate(self.paths_H):
#            if 'face' in v:
#                del self.paths_H[n]
#        time.sleep(1)
        assert self.path, 'Error: H path is empty.'

    def __getitem__(self, index):

        L_path = None

        if self.training == True:
        # ------------------------------------
        # get L image
        # ------------------------------------
            real_L_path_256 = self.real_path_256[index]
            self.name = real_L_path_256.split("/")[-1]
            img_L_r_256 = util.imread_uint(real_L_path_256, self.n_channels)
            H1, W1, C1 = img_L_r_256.shape
            # real_L_path_512 = self.real_path_512[index]
            # self.name = real_L_path_512.split("/")[-1]
            # img_L_r_512 = util.imread_uint(real_L_path_512, self.n_channels)
            # H2, W2, C2 = img_L_r_512.shape
            # rnd_h1_H = random.randint(0, max(0, H1 - self.lq_patchsize))
            # rnd_w1_H = random.randint(0, max(0, W1 - self.lq_patchsize))


        # ------------------------------------
        # get H image
        # ------------------------------------
            H_path = self.path[index]

            img_H = util.imread_uint(H_path, self.n_channels)

            img_name, ext = os.path.splitext(os.path.basename(H_path))
            H, W, C = img_H.shape

            if H < self.patch_size or W < self.patch_size:
                img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8), (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            rnd_h_H1 = random.randint(0, max(0, H - self.lq_patchsize))
            rnd_w_H1 = random.randint(0, max(0, W - self.lq_patchsize))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
            img_L_r_256 = img_L_r_256[int(rnd_h_H/2):int(rnd_h_H/2) + self.lq_patchsize, int(rnd_w_H/2):int(rnd_w_H/2) + self.lq_patchsize, :]
            # img_L_r_512 = img_L_r_512[int(rnd_h_H1):int(rnd_h_H1) + self.lq_patchsize, int(rnd_w_H1):int(rnd_w_H1) + self.lq_patchsize, :]



            if 'face' in img_name:
                mode = random.choice([0, 4])
                img_H = util.augment_img(img_H, mode=mode)
            else:
                mode = random.randint(0, 7)

                img_H = util.augment_img(img_H, mode=mode)
                img_L_r_256 = util.augment_img(img_L_r_256, mode=mode)
                # img_L_r_512 = util.augment_img(img_L_r_512, mode=mode)

            img_H = util.uint2single(img_H)
            # if self.degradation_type == 'bsrgan':
            img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize, isp_model=None)
            # elif self.degradation_type == 'bsrgan_plus':
            #     img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
            img_L_r_256 = util.uint2single(img_L_r_256)
            img_L_r_256 = util.single2tensor3(img_L_r_256)

            # img_L_r_512 = util.uint2single(img_L_r_512)
            # img_L_r_512 = util.single2tensor3(img_L_r_512)

            # if L_path is None:
            #   L_path = H_path

            return img_L, img_H, img_L_r_256

        else:
            # test_L_path_256 = self.real_path_256[index]
            # self.name = test_L_path_256.split("/")[-1]
            # img_L_test_256 = util.imread_uint(test_L_path_256, self.n_channels)
            # img_L_test_256 = util.uint2single(img_L_test_256)
            # img_L_test_256 = util.single2tensor3(img_L_test_256)
            test_L_path_512 = self.real_path_512[index]
            self.name = test_L_path_512.split("/")[-1]
            img_L_test_512 = util.imread_uint(test_L_path_512, self.n_channels)
            img_L_test_512 = util.uint2single(img_L_test_512)
            img_L_test_512 = util.single2tensor3(img_L_test_512)

            return img_L_test_512

            # elif self.degradation_type == 'bsrgan_plus':
            #     img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob, use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------

    def __len__(self):
        if self.training:
            return len(self.real_path_256)
        else:
            return len(self.real_path_512)

    def getFileName(self):
        return self.name
