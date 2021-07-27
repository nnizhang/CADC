from ImageBranchEncoder import ImageBranchEncoder
from ImageBranchDecoder import ImageBranchDecoder
import torch.nn as nn
import torch
from torch.nn import BatchNorm2d as bn
from Modules import KernelModule
from Modules import Transformer
from Modules import poolingModule


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate):
        super(_DenseAsppBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.bn1 = bn(num1, momentum=0.0003)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                               dilation=dilation_rate, padding=dilation_rate)
        self.bn2 = bn(num2, momentum=0.0003)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):

        feature = self.relu1(self.bn1(self.conv1(input)))
        feature = self.relu2(self.bn2(self.conv2(feature)))

        return feature


class DASPPmodule(nn.Module):
    def __init__(self):
        super(DASPPmodule, self).__init__()
        num_features = 512
        d_feature1 = 176
        d_feature0 = num_features//2

        self.AvgPool = nn.Sequential(
            nn.AvgPool2d([32, 32], [32, 32]),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(size=32, mode='nearest'),
        )
        self.ASPP_2 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=2)

        self.ASPP_4 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=4)

        self.ASPP_8 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=8)

        self.afterASPP = nn.Sequential(
            nn.Conv2d(in_channels=512*2 + 176*3, out_channels=512, kernel_size=1),)

    def forward(self, encoder_fea):

        imgAvgPool = self.AvgPool(encoder_fea)

        aspp2 = self.ASPP_2(encoder_fea)
        feature = torch.cat([aspp2, encoder_fea], dim=1)

        aspp4 = self.ASPP_4(feature)
        feature = torch.cat([aspp4, feature], dim=1)

        aspp8 = self.ASPP_8(feature)
        feature = torch.cat([aspp8, feature], dim=1)

        asppFea = torch.cat([feature, imgAvgPool], dim=1)
        AfterASPP = self.afterASPP(asppFea)

        return AfterASPP


class CoSODNet(nn.Module):
    def __init__(self, n_channels, mode='train'):
        super(CoSODNet, self).__init__()

        self.mode = mode
        self.ImageBranchEncoder = ImageBranchEncoder(n_channels)

        self.ImageBranch_fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.ImageBranch_DASPP = DASPPmodule()

        self.poolingModule = poolingModule()
        self.Transformer = Transformer(512)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.KernelModule = KernelModule()

        self.ImageBranchDecoder = ImageBranchDecoder()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight),
                nn.init.constant_(m.bias, 0),

    def forward(self, image_Input):
        if self.mode == 'train':
            preds = self._train_forward(image_Input)
        else:
            preds = self._test_forward(image_Input)

        return preds

    def _train_forward(self, image_Input):
        outputs_image = self._forward(image_Input)
        return outputs_image

    def _test_forward(self, image_Input):

        with torch.no_grad():
            outputs_image = self._forward(image_Input)
            return outputs_image

    def _forward(self, image_Input):
        N, _, _, _ = image_Input.size()
        image_feas = self.ImageBranchEncoder(image_Input)
        afteraspp = self.ImageBranch_DASPP(self.ImageBranch_fc7_1(image_feas[-1]))

        ####################### Consensus Feature Aggregation #######################
        output = self.poolingModule(afteraspp)  # [N, C, 46]
        Transformer_output, affinity = self.Transformer(output)
        # Transformer_output [N, 46, C]   affinity (Nx46)x(Nx46)
        Transformer_output = self.bn_relu(Transformer_output.permute(0, 2, 1))  # [N, C, 46]

        _, cha, _ = output.size()

        ####################### Consensus-aware Kernel Construction and Search #######################
        kenerled_afteraspp = self.KernelModule(Transformer_output, afteraspp, affinity)

        outputs_image = self.ImageBranchDecoder(image_feas, kenerled_afteraspp, Transformer_output, affinity)

        return outputs_image

    def init_parameters(self, pretrain_vgg16_1024):

        conv_blocks = [self.ImageBranchEncoder.conv1,
                       self.ImageBranchEncoder.conv2,
                       self.ImageBranchEncoder.conv3,
                       self.ImageBranchEncoder.conv4,
                       self.ImageBranchEncoder.conv5,
                       self.ImageBranchEncoder.fc6,
                       self.ImageBranchEncoder.fc7]
        listkey = [['conv1_1', 'conv1_2'], ['conv2_1', 'conv2_2'], ['conv3_1', 'conv3_2', 'conv3_3'],
                   ['conv4_1', 'conv4_2', 'conv4_3'], ['conv5_1', 'conv5_2', 'conv5_3'], ['fc6'], ['fc7']]

        for idx, conv_block in enumerate(conv_blocks):
            num_conv = 0
            for l2 in conv_block:
                if isinstance(l2, nn.Conv2d):
                    num_conv += 1
                    l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.weight']
                    l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.bias'].squeeze(0).squeeze(0).squeeze(0).squeeze(0)
        return self
