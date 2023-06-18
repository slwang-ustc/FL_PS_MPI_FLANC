import math

import torch
from torch import nn

from config import cfg


class DecomBlock(nn.Module):
    """ Decompose a convolution layer to a basis network and a coefficient network """

    def __init__(self, basis_in_channel, basis_out_channel, conv_in_channel, conv_out_channel, stride, bias=False):
        super(DecomBlock, self).__init__()
        self.basis_in_channel = basis_in_channel
        self.group = conv_in_channel / basis_in_channel

        # basis network
        self.basis_conv = nn.Conv2d(
            basis_in_channel, basis_out_channel, kernel_size=3, stride=stride, padding=1, bias=bias,
        )
        # torch.nn.init.orthogonal_(self.basis_conv.weight)
        # coefficients network
        self.coeffi_conv = nn.Conv2d(
            int(self.group * basis_out_channel), conv_out_channel, kernel_size=1, stride=1, bias=bias
        )

    def forward(self, x):
        if self.group == 1:
            x = self.basis_conv(x)
        else:
            x = torch.cat([self.basis_conv(xi) for xi in torch.split(x, self.basis_in_channel, dim=1)], dim=1)
        x = self.coeffi_conv(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, basis_in_channel, basis_out_channel, stride):
        super(BasicBlock, self).__init__()

        self.residual_function = nn.Sequential(
            DecomBlock(basis_in_channel, basis_out_channel, in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes, track_running_stats=False),
            nn.ReLU(inplace=True),
            DecomBlock(basis_in_channel, basis_out_channel, out_planes, out_planes * BasicBlock.expansion, 1),
            nn.BatchNorm2d(out_planes * BasicBlock.expansion, track_running_stats=False)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * BasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes * BasicBlock.expansion, track_running_stats=False)
            )

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        out = residual + shortcut
        out = nn.ReLU(inplace=True)(out)

        return out


class ResNet_Flanc(nn.Module):

    def __init__(self, block, num_block, model_ratio):
        super(ResNet_Flanc, self).__init__()

        self.model_ratio = model_ratio
        self.hidden_size = [64, 64, 128, 256, 512]
        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(cfg['data_shape'][0], self.hidden_size[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.hidden_size[0], track_running_stats=False),
            nn.ReLU(inplace=True)
        )

        modules = []
        for i in range(1, len(self.hidden_size)):
            basis_in_channel1 = round(self.hidden_size[i - 1] * cfg['basis_in_ratio'])
            basis_out_channel = round(self.hidden_size[i] * cfg['basis_out_ratio'])
            basis_in_channel2 = round(self.hidden_size[i] * cfg['basis_in_ratio'])
            modules.append(self._make_layer(
                block, self.hidden_size[i], basis_in_channel1, basis_in_channel2,
                basis_out_channel, num_block[i - 1], 1 if i == 1 else 2
            ))

        self.network = nn.Sequential(*modules)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = int(self.hidden_size[len(self.hidden_size)-1] * model_ratio * block.expansion)
        self.fc = nn.Linear(output_channel, cfg['classes_size'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                for m in [0, 3]:
                    torch.nn.init.orthogonal_(self.network[i][j].residual_function[m].basis_conv.weight)

    def _make_layer(self, block, out_planes, basis_in_channel1, basis_in_channel2,
                    basis_out_channel, num_blocks, stride):
        out_planes = round(out_planes * self.model_ratio)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = [block(self.in_planes, out_planes, basis_in_channel1, basis_out_channel, stride)]
        self.in_planes = out_planes * block.expansion
        for stride in strides[1:]:
            layers.append(block(self.in_planes, out_planes, basis_in_channel2, basis_out_channel, stride))
            self.in_planes = out_planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.network(output)
        output = self.avg_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output


def resnet18_flanc(model_ratio):
    """
    return a ResNet 18 object
    """
    return ResNet_Flanc(BasicBlock, [2, 2, 2, 2], model_ratio)
