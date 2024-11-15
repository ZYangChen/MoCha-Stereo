import torch
import torch.nn as nn
import torch.nn.functional as F
from core.submodule import *

class Mca_Camp(nn.Module):
    def __init__(self):
        super(Mca_Camp, self).__init__()
        chans = [256,192,448,384]
        laten_chans = [4,4,4,4]
        self.sw_4x = nn.Conv2d(chans[0] * 2, laten_chans[0], 3, 1, 1, bias=False)
        self.sw_8x = nn.Conv2d(chans[1] * 2,laten_chans[1], 3, 1, 1, bias=False)
        self.sw_16x = nn.Conv2d(chans[2] * 2,laten_chans[2], 3, 1, 1, bias=False)

        self.Conv3d_layer_4x = BasicConv_IN(laten_chans[0],1, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.Conv3d_layer_8x = BasicConv_IN(laten_chans[1],1, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.Conv3d_layer_16x = BasicConv_IN(laten_chans[2],1, is_3d=True, kernel_size=3, stride=1, padding=1)

    def high_frequency_sliding_window_filter(self, feature):
        B,C,H,W = feature.shape
        kernel_size = 3
        # 创建高斯核
        kernel = torch.ones((C, C, kernel_size, kernel_size)) / (kernel_size ** 2)
        kernel = kernel.to(feature.device)

        # 在输入上应用高斯滤波
        smoothed = F.conv2d(feature, kernel, padding=kernel_size // 2)

        # 高通滤波器: 输入减去平滑后的结果
        # print('f',feature.shape)
        # print('s',smoothed.shape)
        feature = feature - smoothed

        return feature


    def forward(self, features_left, features_right):

        normal_feature= []
        normal_feature.append(torch.cat([features_left[0], features_right[0]], dim=1))
        normal_feature.append(torch.cat([features_left[1], features_right[1]], dim=1))
        normal_feature.append(torch.cat([features_left[2], features_right[2]], dim=1))

        #mca 1/4
        normal_feature[0] = self.high_frequency_sliding_window_filter(normal_feature[0])
        motif_4x = F.softmax(self.sw_4x(normal_feature[0]), dim=1)#torch.Size([B, 4, 80, 184])
        #camp
        CAMP_4x = motif_4x.unsqueeze(1) * features_left[0].unsqueeze(2) #torch.Size([B, 256, 4, 80, 184])
        CAMP_4x = CAMP_4x.transpose(1, 2) #torch.Size([B, 4, 256, 80, 184])
        #Conv3d_layer
        channel_correlation_volume_4x = self.Conv3d_layer_4x(CAMP_4x).squeeze(1)

        #mca 1/8
        normal_feature[1] = self.high_frequency_sliding_window_filter(normal_feature[1])
        motif_8x = F.softmax(self.sw_8x(normal_feature[1]), dim=1)
        # camp
        CAMP_8x = motif_8x.unsqueeze(1) * features_left[1].unsqueeze(2)
        CAMP_8x = CAMP_8x.transpose(1, 2)
        # Conv3d_layer
        channel_correlation_volume_8x = self.Conv3d_layer_8x(CAMP_8x).squeeze(1)

        # mca 1/16
        normal_feature[2] = self.high_frequency_sliding_window_filter(normal_feature[2])
        motif_16x = F.softmax(self.sw_16x(normal_feature[2]), dim=1)
        # camp
        CAMP_16x = motif_16x.unsqueeze(1) * features_left[2].unsqueeze(2)
        CAMP_16x = CAMP_16x.transpose(1, 2)
        # Conv3d_layer
        channel_correlation_volume_16x = self.Conv3d_layer_16x(CAMP_16x).squeeze(1)


        return [channel_correlation_volume_4x, channel_correlation_volume_8x, channel_correlation_volume_16x, features_left[3]]


