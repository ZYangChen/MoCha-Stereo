import torch
import torch.nn as nn
import torch.nn.functional as F
from core.submodule import *
import math
# import pywt
from pytorch_wavelets import DWTForward, DWTInverse
from core.motif import image_motif

class wavelet_transform(nn.Module):
    def __init__(self):
        super(wavelet_transform, self).__init__()
        self.xfm = DWTForward(J=3, mode='zero', wave='haar')
        self.motif = image_motif()

    def forward(self, LL):
        LL, HH = self.xfm(LL)
        # print('LL',LL.shape)
        LH = HH[0]
        # print('LH',HH[0].shape)
        HL = HH[1]
        # print('HL',HH[1].shape)
        HH = HH[2]
        # print('HH',HH[2].shape)
        DWTmap = [None, None, None]
        for i in range(3):
            if i == 0:
                val = LH
                # continue
            elif i == 1:
                val = HL
                # continue
            else:
                val = HH
                # print('HH')
            b, c, n, h, w = val.shape
            # print('n',n)
            val = val.view(b, c, n * h, w)

            val = val.chunk(3, dim=2)
            H = self.motif(val[0]).unsqueeze(2)
            V = self.motif(val[1]).unsqueeze(2)
            D = self.motif(val[2]).unsqueeze(2)
            val = torch.cat((H,V,D),dim=2)
            DWTmap[i] = val

        return self.motif(LL), DWTmap

class wavelet_inverse_transform(nn.Module):
    def __init__(self):
        super(wavelet_inverse_transform, self).__init__()
        self.ifm = DWTInverse(mode='zero', wave='haar')

    def forward(self, LL, DWTmap):
        LL = self.ifm((LL, DWTmap))

        return LL

class motif_dist(nn.Module):
    def __init__(self):
        super(motif_dist, self).__init__()
        self.unfold = nn.Unfold(kernel_size=3)
        self.group = 16

    def forward(self, feature):
        _, C, H, W = feature.shape
        fold = nn.Fold(output_size=(H,W), kernel_size=3)
        feature = feature.chunk(self.group, dim=1)
        for i in range(self.group):
            series = self.unfold(feature[i])
            ser = series.chunk(9, dim=1)
            for j in range(9):
                motif = ser[j]
                distances = torch.cdist(motif, motif).cuda()
                b, width, height = distances.shape
                for k in range(b):
                    diagonal_indices = torch.arange(min(width, height))
                    distances[k,diagonal_indices,diagonal_indices]=float('inf')

                min_indices = torch.argmin(distances, dim=-1)

                for l in range(b):
                    unique, counts = torch.unique(min_indices[l], return_counts=True)

                    for o in range(len(unique)):
                        if l == 0:
                            motif_new = motif[l, :, :] * motif[l, unique[o], :] * counts[o] // (C // self.group)
                        else:
                            motif_new = motif[l, :, :] + motif[l, :, :] * motif[l, unique[o], :] * counts[o] // (C // self.group)

                    if l == 0:
                        motif_cat = motif_new.unsqueeze(0)
                    else:
                        motif_cat = torch.cat((motif_cat, motif_new.unsqueeze(0)), dim=0)


                if j == 0:
                    series = motif_cat
                else:
                    series = torch.cat((series, motif_cat), dim=1)



            series = fold(series)

            if i == 0:
                motif_feature = series
            else:
                motif_feature = torch.cat((motif_feature, series), dim=1)


        return motif_feature

class DCT_motif(nn.Module):
    def __init__(self):
        super(DCT_motif, self).__init__()

        self.wavelet_transform = wavelet_transform()

        self.inverse = wavelet_inverse_transform()

    def forward(self, LL):
        LL, DWTmap = self.wavelet_transform(LL)
        LL = self.inverse(LL, DWTmap)

        return LL


class TwoViewFeature(nn.Module):
    def __init__(self, mixed_precision):
        super(TwoViewFeature, self).__init__()
        self.mixed_precision = mixed_precision
        chans = [256,192,448,384]
        laten_chans = 8

        self.DM_4x = DCT_motif()
        self.Conv_4x = nn.Conv2d(in_channels=chans[0]*2, out_channels=chans[0], kernel_size=1, stride=1, padding=0)
        self.DM_8x = DCT_motif(chans[1]*2, laten_chans)
        self.Conv_8x = nn.Conv2d(in_channels=chans[1]*2, out_channels=chans[1], kernel_size=1, stride=1, padding=0)
        self.DM_16x = DCT_motif(chans[2]*2, laten_chans)
        self.Conv_16x = nn.Conv2d(in_channels=chans[2]*2, out_channels=chans[2], kernel_size=1, stride=1, padding=0)


    def forward(self, features_left, features_right):
        normal_channel_feature = []
        normal_channel_feature.append(torch.cat([features_left[0], features_right[0]], dim = 1))
        normal_channel_feature.append(torch.cat([features_left[1], features_right[1]], dim = 1))
        normal_channel_feature.append(torch.cat([features_left[2], features_right[2]], dim = 1))

        #TODO 1/4
        normal_channel_feature_4x = self.DM_4x(normal_channel_feature[0])
        if self.mixed_precision:
            normal_channel_feature_4x = normal_channel_feature_4x.to(torch.float16)
        normal_channel_feature_4x = self.Conv_4x(normal_channel_feature_4x)
        # print('succeed')

        #TODO 1/8
        normal_channel_feature_8x = self.DM_8x(normal_channel_feature[1])
        if self.mixed_precision:
            normal_channel_feature_8x = normal_channel_feature_8x.to(torch.float16)
        normal_channel_feature_8x = self.Conv_8x(normal_channel_feature_8x)


        # TODO 1/16

        normal_channel_feature_16x = self.DM_16x(normal_channel_feature[2])
        if self.mixed_precision:
            normal_channel_feature_16x = normal_channel_feature_16x.to(torch.float16)
        normal_channel_feature_16x = self.Conv_16x(normal_channel_feature_16x)

        # TODO 1/32
        normal_channel_feature_32x = features_left[3]

        return [normal_channel_feature_4x, normal_channel_feature_8x, normal_channel_feature_16x, normal_channel_feature_32x]