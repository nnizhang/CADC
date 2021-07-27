import torch.nn as nn
import torch
import torch.nn.functional as F


class KernelModule(nn.Module):
    def __init__(self, channel=512):
        super(KernelModule, self).__init__()

        self.encoder_fea_channel = channel
        self.self_att = nn.Sequential(
            nn.Linear(46 * self.encoder_fea_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 46)
        )

        self.generate_depthwise_Adakernel = nn.Sequential(
            nn.Linear(46, 46),
            nn.BatchNorm1d(46),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(46, 9)
        )

        self.generate_pointwise_Adakernel = nn.Sequential(
            nn.Linear(self.encoder_fea_channel, self.encoder_fea_channel),
            nn.BatchNorm1d(self.encoder_fea_channel),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(self.encoder_fea_channel, self.encoder_fea_channel * 64)
        )

        self.att_depthwise_part1 = nn.Sequential(
            nn.Linear(46 * self.encoder_fea_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.encoder_fea_channel)
        )

        self.att_depthwise_part2 = nn.Sequential(
            nn.Linear(46 * self.encoder_fea_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 46)
        )

        self.generate_depthwise_Cokernel = nn.Sequential(
            nn.Linear(46, 46),
            nn.BatchNorm1d(46),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(46, 9)
        )

        self.generate_pointwise_Cokernel = nn.Sequential(
            nn.Linear(self.encoder_fea_channel, self.encoder_fea_channel),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(self.encoder_fea_channel, self.encoder_fea_channel * 64)
        )

    def forward(self, output, enc_fea, affinity):

        N, cha, _ = output.size()

        ##################### Adaptive Kernel Construction #####################
        # [1] depthwise adaptive kernels
        depthwise_Adakernel = self.generate_depthwise_Adakernel(output.view(N * cha, 46))
        depthwise_Adakernel = depthwise_Adakernel.reshape(N, cha, 3, 3)
        depthwise_Adakernel = depthwise_Adakernel.unsqueeze(2)

        # [2] pointwise adaptive kernels
        att = F.softmax(self.self_att(output.view(-1, 46 * cha)), dim=1)  # N x 46
        att = att.unsqueeze(2) # N x 46 x 1
        fea = ((output.permute(0, 2, 1)) * att).sum(dim=1).reshape(N, cha) # N x 512
        pointwise_Adakernel = self.generate_pointwise_Adakernel(fea)
        pointwise_Adakernel = pointwise_Adakernel.reshape(N, 64, cha, 1, 1)


        ##################### Common Kernel Construction #####################
        # [1] pointwise common kernel
        affinity_att = affinity.mean(dim=0)  # [N*46]
        fea_for_pointwise = torch.matmul(affinity_att.unsqueeze(0),
                                         output.permute(0, 2, 1).contiguous().view(N * 46, cha))  # [1, C]
        pointwise_Cokernel = self.generate_pointwise_Cokernel(fea_for_pointwise)
        pointwise_Cokernel = pointwise_Cokernel.reshape(64, cha, 1, 1)

        # [2] depthwise common kernel
        att_part1_depthwise = self.att_depthwise_part1(output.permute(0, 2, 1).contiguous().view(N, 46 * cha))
        att_part1_depthwise = F.softmax(att_part1_depthwise, dim=1) # alpha1 [N, C]

        att_part2_depthwise = self.att_depthwise_part2(output.permute(0, 2, 1).contiguous().view(N, 46 * cha))
        att_part2_depthwise = F.softmax(att_part2_depthwise, dim=1)  # alpha2 [N, 46]

        att_for_depthwise = (output * att_part1_depthwise.unsqueeze(2)).sum(dim=1) # [N, 46]
        att_for_depthwise = (att_for_depthwise * att_part2_depthwise).sum(dim=1) # [N]
        att_for_depthwise = F.softmax(att_for_depthwise).unsqueeze(0)  # alpha3 [1 x N]

        # att_for_depthwise [1 x N]   output [N, C, 46]
        fea_for_depthwise = torch.matmul(att_for_depthwise, output.view(N, cha * 46))  # [1, C*46]

        depthwise_Cokernel = self.generate_depthwise_Cokernel(fea_for_depthwise.reshape(cha, 46))
        depthwise_Cokernel = depthwise_Cokernel.reshape(cha, 3, 3)
        depthwise_Cokernel = depthwise_Cokernel.unsqueeze(1)


        ##################### Searching via Adaptive Kernel #####################
        _, _, H, W = enc_fea.size()

        Adpkenerled_enc_fea = torch.cuda.FloatTensor(N, 64, H, W)

        for num in range(N):
            tmp_fea = F.conv2d(enc_fea[num, :, :, :].unsqueeze(0), depthwise_Adakernel[num, :, :, :, :], stride=1,
                               padding=1, groups=cha)
            Adpkenerled_enc_fea[num, :, :, :] = F.conv2d(tmp_fea, pointwise_Adakernel[num, :, :, :, :], stride=1,
                                                           padding=0)

        ##################### Searching via Common Kernel #####################
        Cokenerled_enc_fea = F.conv2d(enc_fea, depthwise_Cokernel, stride=1, padding=1, groups=cha)
        Cokenerled_enc_fea = F.conv2d(Cokenerled_enc_fea, pointwise_Cokernel, stride=1, padding=0)

        kenerled_afteraspp = torch.cat([Adpkenerled_enc_fea, Cokenerled_enc_fea], dim=1)

        return kenerled_afteraspp


class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)

        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, ori_feature):
        # ori_feature N x C x 46
        feature = self.bn_relu(ori_feature)
        feature = feature.permute(0, 2, 1)

        # feature N x 46 x C
        N, num, c = feature.size()

        x_theta = self.theta(feature.contiguous().view(-1, c))
        x_phi = self.phi(feature.contiguous().view(-1, c))
        x_phi = x_phi.permute(1, 0)
        attention = torch.matmul(x_theta, x_phi)

        for k in range(N):
            attention[k*46:k*46+46, k*46:k*46+46] = -1000

        # (Nx46)x(Nx46)
        f_div_C = F.softmax(attention, dim=-1)

        g_x = self.g((feature.contiguous().view(-1, c)))
        y = torch.matmul(f_div_C, g_x)
        # (Nx46)xc/2

        W_y = self.W(y).contiguous().view(N, num, c)

        att_fea = ori_feature.permute(0, 2, 1) + W_y

        return att_fea, f_div_C


class poolingModule(nn.Module):
    def __init__(self):
        super(poolingModule, self).__init__()

        self.maxpool1 = torch.nn.AdaptiveMaxPool2d((1,1))
        self.maxpool2 = torch.nn.AdaptiveMaxPool2d((3,3))
        self.maxpool3 = torch.nn.AdaptiveMaxPool2d((6,6))

    def forward(self, feature):
        batch_size, cha, _, _ = feature.size()
        maxpool_fea1 = self.maxpool1(feature).view(batch_size, cha, -1)
        maxpool_fea2 = self.maxpool2(feature).view(batch_size, cha, -1)
        maxpool_fea3 = self.maxpool3(feature).view(batch_size, cha, -1)

        maxpool_fea = torch.cat([maxpool_fea1, maxpool_fea2, maxpool_fea3], dim=2)

        return maxpool_fea

