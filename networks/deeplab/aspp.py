import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, pretrained):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight(pretrained)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self,pretrained):
        for m in self.modules():
            if pretrained:
                break
            else:

                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, pretrained, inplanes=2048, outplanes = 256):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, outplanes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm, pretrained=pretrained)
        self.aspp2 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm, pretrained=pretrained)
        self.aspp3 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm, pretrained=pretrained)
        self.aspp4 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm, pretrained=pretrained)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             BatchNorm(outplanes),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(outplanes*5, outplanes, 1, bias=False)
        self.bn1 = BatchNorm(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight(pretrained)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        # if type(x4.size()[2]) != int:
        #     tmpsize = (x4.size()[2].item(),x4.size()[3].item())
        # else:
        #     tmpsize = (x4.size()[2],x4.size()[3])
        # x5 = F.interpolate(x5, size=(14,14), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self,pretrained):
        for m in self.modules():
            if pretrained:
                break
            else:
                if isinstance(m, nn.Conv2d):
                    # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    # m.weight.data.normal_(0, math.sqrt(2. / n))
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm,pretrained):
    return ASPP(backbone, output_stride, BatchNorm, pretrained)