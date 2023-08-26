from torch import nn

import torch
import torch.nn as nn

from models.SSEN import SSEN,SSEN_show
from models.Model_utils import make_residual_block, make_downsampling_network
from models.utils import showpatch

class Similar_domain(nn.Module):
    def __init__(self, num_channel = 64, mode = "add"):
        super(Similar_domain, self).__init__()
        # referenced by EDVR paper implementation code
        # https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py line 251
        self.downsampling_network = make_downsampling_network(layernum=2, in_channels=1, out_channels=64)
        self.lrfeature_scaler = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, bias=False)
        self.feature_extractor = make_residual_block(blocknum=5, input_channel=64, output_channel=64)

        self.SSEN_Network = SSEN(in_channels=num_channel,mode = mode)
        self.SSEN_Network_inverse = SSEN(in_channels=num_channel, mode=mode)

        self.preprocessing_residual_block = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.residual_blocks = make_residual_block(blocknum=16, input_channel=64, output_channel=64)

        self.upscaling_2x = nn.Sequential(
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            nn.PixelShuffle(2),
            # nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            # nn.PixelShuffle(2),
        )

        self.outconv = nn.Conv2d(in_channels=num_channel, out_channels=1, kernel_size=3, padding=1, bias=False)


    def forward(self,input_lr, ref_input):
        ref_input = self.downsampling_network(ref_input)
        input_lr = self.lrfeature_scaler(input_lr)
        lr_feature_out = self.feature_extractor(input_lr)
        ref_feature_out = self.feature_extractor(ref_input)

        # SSEN_out = self.SSEN_Network(lr_batch = lr_feature_out ,init_hr_batch = ref_feature_out)
        SSEN_out_inverse = self.SSEN_Network_inverse(lr_batch=ref_feature_out, init_hr_batch=lr_feature_out)
        # residual_input = torch.cat((lr_feature_out, SSEN_out), dim = 1)
        # residual_input_scaled = self.preprocessing_residual_block(residual_input)
        # lr---sr
        out_sr = self.residual_blocks(lr_feature_out)
        out_sr = torch.add(out_sr,lr_feature_out)

        out_sr = self.upscaling_2x(out_sr)
        out_sr = self.outconv(out_sr)
        # lr_pair---sr_pair
        out_sr_pair = self.residual_blocks(SSEN_out_inverse)
        out_sr_pair = torch.add(out_sr_pair, SSEN_out_inverse)

        out_sr_pair = self.upscaling_2x(out_sr_pair)
        out_sr_pair = self.outconv(out_sr_pair)
        # lr_real---sr_real
        out_sr_real = self.residual_blocks(ref_feature_out)
        out_sr_real = torch.add(out_sr_real, ref_feature_out)

        out_sr_real = self.upscaling_2x(out_sr_real)
        out_sr_real = self.outconv(out_sr_real)

        return out_sr,out_sr_pair,out_sr_real


class Similar_domain_show(nn.Module):
    def __init__(self, num_channel = 64, mode = "add"):
        super(Similar_domain_show, self).__init__()
        # referenced by EDVR paper implementation code
        # https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py line 251
        self.downsampling_network = make_downsampling_network(layernum=2, in_channels=1, out_channels=64)
        self.lrfeature_scaler = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, bias=False)
        self.feature_extractor = make_residual_block(blocknum=5, input_channel=64, output_channel=64)

        self.SSEN_Network = SSEN(in_channels=num_channel, mode=mode)

        self.preprocessing_residual_block = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.residual_blocks = make_residual_block(blocknum=16, input_channel=64, output_channel=64)

        self.upscaling_2x = nn.Sequential(
            nn.Conv2d(in_channels=num_channel, out_channels=4 * num_channel, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            # nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            # nn.PixelShuffle(2),
        )

        self.outconv = nn.Conv2d(in_channels=num_channel, out_channels=1, kernel_size=3, padding=1, bias=False)


    def forward(self,input_lr, ref_input , showmode = False):
        ref_input = self.downsampling_network(ref_input)
        input_lr = self.lrfeature_scaler(input_lr)
        lr_feature_out = self.feature_extractor(input_lr)
        ref_feature_out = self.feature_extractor(ref_input)

        SSEN_out = self.SSEN_Network(lr_batch = lr_feature_out ,init_hr_batch = ref_feature_out, showmode=showmode)
        residual_input = torch.cat((lr_feature_out, SSEN_out), dim = 1)
        residual_input_scaled = self.preprocessing_residual_block(residual_input)
        out = self.residual_blocks(residual_input_scaled)
        out = torch.add(out,lr_feature_out)

        if showmode:
            showpatch(lr_feature_out,foldername="extracted_features_lr_image")
            showpatch(ref_feature_out,foldername="extracted_features_ref_image")
            showpatch(out, foldername="features_after_reconstruction_blocks")

        out = self.upscaling_4x(out)
        out = self.outconv(out)
        return out