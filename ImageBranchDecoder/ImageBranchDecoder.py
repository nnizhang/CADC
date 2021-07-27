import torch.nn as nn
import torch
import torch.nn.functional as F
from parameter import *
from Modules import KernelModule
from Modules import Transformer
from Modules import poolingModule


class spatialAttDecoder_module(nn.Module):
    def __init__(self, in_channels, out_channels, fusing=True):
        super(spatialAttDecoder_module, self).__init__()
        if fusing:
            self.enc_fea_proc = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            )
            in_channels = in_channels + dec_channels
        self.decoding1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.decoding2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.SpatialAtt = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, enc_fea, dec_fea=None):
        if dec_fea is not None:
            enc_fea = self.enc_fea_proc(enc_fea)
            if dec_fea.size(2) != enc_fea.size(2):
                dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear', align_corners=True)

            spatial_att = self.SpatialAtt(dec_fea)
            enc_fea = enc_fea*spatial_att
            enc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        output = self.decoding1(enc_fea)
        output = self.decoding2(output)

        return output


class decoder_module(nn.Module):
    def __init__(self, in_channels, out_channels, fusing=True):
        super(decoder_module, self).__init__()
        if fusing:
            self.enc_fea_proc = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            )
            self.poolingModule = poolingModule()
            self.Transformer = Transformer(in_channels)

            self.bn_relu = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
            )
            self.KernelModule = KernelModule(in_channels)

            in_channels = 3*dec_channels

        self.decoding1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.decoding2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc_fea, dec_fea=None):
        if dec_fea is not None:

            # [1] Consensus Feature Aggregation
            output = self.poolingModule(enc_fea) # [N, C, 46]
            Transformer_output, affinity = self.Transformer(output)
            # Transformer_output [N, 46, C]   affinity (Nx46)x(Nx46)
            Transformer_output = self.bn_relu(Transformer_output.permute(0, 2, 1)) # [N, C, 46]

            # [2] Consensus-aware Kernel Construction and Search
            enc_fea = self.enc_fea_proc(enc_fea)
            enc_fea = self.KernelModule(Transformer_output, enc_fea, affinity)

            if dec_fea.size(2) != enc_fea.size(2):
                dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear', align_corners=True)
            enc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        output = self.decoding1(enc_fea)
        output = self.decoding2(output)

        return output


class ImageBranchDecoder(nn.Module):
    def __init__(self):

        super(ImageBranchDecoder, self).__init__()
        channels = [64, 128, 256, 512, 512, 512]

        self.decoder6 = decoder_module(dec_channels*2, dec_channels, False)
        self.decoder5 = decoder_module(channels[4], dec_channels)
        self.decoder4 = decoder_module(channels[3], dec_channels)
        self.decoder3 = decoder_module(channels[2], dec_channels)
        self.decoder2 = spatialAttDecoder_module(channels[1], dec_channels)
        self.decoder1 = spatialAttDecoder_module(channels[0], dec_channels)

        self.conv_loss6 = nn.Conv2d(in_channels=dec_channels, out_channels=1, kernel_size=3, padding=1)
        self.conv_loss5 = nn.Conv2d(in_channels=dec_channels, out_channels=1, kernel_size=3, padding=1)
        self.conv_loss4 = nn.Conv2d(in_channels=dec_channels, out_channels=1, kernel_size=3, padding=1)
        self.conv_loss3 = nn.Conv2d(in_channels=dec_channels, out_channels=1, kernel_size=3, padding=1)
        self.conv_loss2 = nn.Conv2d(in_channels=dec_channels, out_channels=1, kernel_size=3, padding=1)
        self.conv_loss1 = nn.Conv2d(in_channels=dec_channels, out_channels=1, kernel_size=3, padding=1)

    def forward(self, enc_fea, kenerled_afteraspp, Transformer_output, affinity):

        encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4, encoder_conv5, x7 = enc_fea

        dec_fea_6 = self.decoder6(kenerled_afteraspp)
        mask6 = self.conv_loss6(dec_fea_6)

        dec_fea_5 = self.decoder5(encoder_conv5, dec_fea_6)
        mask5 = self.conv_loss5(dec_fea_5)

        dec_fea_4 = self.decoder4(encoder_conv4, dec_fea_5)
        mask4 = self.conv_loss4(dec_fea_4)

        dec_fea_3 = self.decoder3(encoder_conv3, dec_fea_4)
        mask3 = self.conv_loss3(dec_fea_3)

        dec_fea_2 = self.decoder2(encoder_conv2, dec_fea_3)
        mask2 = self.conv_loss2(dec_fea_2)

        dec_fea_1 = self.decoder1(encoder_conv1, dec_fea_2)
        mask1 = self.conv_loss1(dec_fea_1)

        return mask6, mask5, mask4, mask3, mask2, mask1
