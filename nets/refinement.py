import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.feature import BasicBlock, BasicConv, Conv2x
from nets.warp import disp_warp

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))

class FeatureAtt(nn.Module):
    def __init__(self, in_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(in_chan, in_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_chan//2, in_chan, 1))

    def forward(self, feat):
        feat_att = self.feat_att(feat)
        '''
        https://blog.csdn.net/amusi1994/article/details/122315076
        在使用ce loss 或者 bceloss的时候，会有log的操作，在半精度情况下，一些非常小的数值会被直接舍入到0，log(0)等于nan
        于是回传的梯度因为log而变为nan->网络参数nan-> 每轮输出都变成nan
        '''
        feat_att = feat_att.float()
        feat = torch.sigmoid(feat_att)*feat
        return feat

class Attention_HourglassModel(nn.Module):
    def __init__(self, in_channels):
        super(Attention_HourglassModel, self).__init__()
        self.conv1a = BasicConv(in_channels, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, in_channels, deconv=True)
        #Attention
        self.feature_att_2 = FeatureAtt(48)
        self.feature_att_4 = FeatureAtt(64)
        self.feature_att_8 = FeatureAtt(96)
        self.feature_att_16 = FeatureAtt(128)

    def forward(self, x):
        rem0 = x
        x = self.conv1a(x)
        x = self.feature_att_2(x)
        rem1 = x
        x = self.conv2a(x)
        x = self.feature_att_4(x)
        rem2 = x
        x = self.conv3a(x)
        x = self.feature_att_8(x)
        rem3 = x
        x = self.conv4a(x)
        x = self.feature_att_16(x)
        rem4 = x

        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        return x

class Simple_UNet(nn.Module):
    def __init__(self, in_channels):
        super(Simple_UNet, self).__init__()
        self.conv1a = BasicConv(in_channels, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, in_channels, deconv=True)

    def forward(self, x):
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x

        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        return x


from PIL import Image
import numpy as np
def save_feature_map_as_image(feature_map, file_path):
    # 假设 feature_map 是一个形状为 torch.Size([1, 3, 1952, 2880]) 的 PyTorch 张量
    feature_map = feature_map.squeeze(0)  # 去除批量维度，如果有的话

    # 将张量转换为 NumPy 数组并交换轴，变为形状 [H, W, 3]
    feature_map = feature_map.permute(1, 2, 0).cpu().numpy()

    # 将数值范围缩放到 0 到 255 之间
    feature_map = ((feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255).astype('uint8')

    # 创建 PIL 图像对象
    image = Image.fromarray(feature_map)

    # 保存为 PNG 图像
    image.save(file_path)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class REMP(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(REMP, self).__init__()

        # Left and warped flaw
        in_channels = 6
        channel =32
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.conv_start = BasicConv(32, channel, kernel_size=3, padding=2, dilation=2)

        self.RefinementBlock = Simple_UNet(in_channels=channel)#, in_channels

        self.AP = nn.AdaptiveAvgPool2d(1)
        self.LFE = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 2, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.LMC = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel * 2, 3),
            nn.ReLU(inplace=True),
            default_conv(channel * 2, channel, 3),
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):

        assert low_disp.dim() == 4
        low_disp = -low_disp
        # low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor
        del low_disp
        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, 3, H, W]
        # save_feature_map_as_image(warped_right, 'WR.png')
        flaw = warped_right - left_img  # [B, 3, H, W]
        # print('flaw',flaw.shape)
        # save_feature_map_as_image(flaw, 'RE.png')
        del warped_right
        ref_flaw = torch.cat((flaw, left_img), dim=1)  # [B, 6, H, W]
        del flaw
        ref_flaw = self.conv1(ref_flaw)  # [B, 16, H, W]
        disp_fea = self.conv2(disp)  # [B, 16, H, W]
        x = torch.cat((ref_flaw, disp_fea), dim=1)  # [B, 32, H, W]
        del ref_flaw, disp_fea
        x = self.conv_start(x)  # [B, 32, H, W]

        x = self.RefinementBlock(x) # [B, 32, H, W]

        low = self.LFE(self.AP(x))
        motif = self.LMC(x)
        x = torch.mul((1 - motif), low) + torch.mul(motif, x)

        x = self.final_conv(x)  # [B, 1, H, W]

        disp = F.relu(disp + x, inplace=True)  # [B, 1, H, W]
        # warped_right = disp_warp(right_img, new_disp)[0]  # [B, C, H, W]
        # new_flaw = warped_right - left_img  # [B, C, H, W]
        # new_flaw = torch.sum(new_flaw, dim=1)
        # flaw = torch.sum(flaw, dim=1)
        # # print(new_disp.shape)
        # B, _, H, W = left_img.shape
        # disp = torch.where((new_flaw < flaw).view(B, 1, H, W),
        #                    new_disp, disp)

        return -disp
