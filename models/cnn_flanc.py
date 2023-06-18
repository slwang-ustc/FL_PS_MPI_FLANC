from config import cfg
import torch.nn as nn
import torch


class DecomBlock(nn.Module):
    """ Decompose a convolution layer to a basis network and a coefficient network """
    def __init__(self, basis_in_channel, basis_out_channel, conv_in_channel, conv_out_channel, bias=False):
        super(DecomBlock, self).__init__()
        self.basis_in_channel = basis_in_channel
        self.group = conv_in_channel / basis_in_channel

        # basis network
        self.basis_conv = nn.Conv2d(
            basis_in_channel, basis_out_channel, bias=bias, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.orthogonal_(self.basis_conv.weight)
        # coefficients network
        self.coeffi_conv = nn.Conv2d(
            int(self.group * basis_out_channel), conv_out_channel, bias=bias, kernel_size=1, stride=1
        )

    def forward(self, x):
        if self.group == 1:
            x = self.basis_conv(x)
        else:
            x = torch.cat([self.basis_conv(xi) for xi in torch.split(x, self.basis_in_channel, dim=1)], dim=1)
        x = self.coeffi_conv(x)
        return x


class CNN_FLANC(nn.Module):
    """ Simple convolutional network """

    def __init__(self, model_ratio):
        super().__init__()
        hidden_sizes = cfg['cnn_hidden_size']

        # the head layer
        modules = [nn.Conv2d(cfg['data_shape'][0], hidden_sizes[0], kernel_size=3, stride=1, padding=1)]

        # For each hidden layer, first scale it (similar to HeteroFL) and then decompose it using DecomBlock():
        for i in range(1, len(hidden_sizes)):
            basis_in_channel = round(hidden_sizes[i-1] * cfg['basis_in_ratio'])
            basis_out_channel = round(hidden_sizes[i] * cfg['basis_out_ratio'])
            if i == 1:
                conv_in_channel = hidden_sizes[0]
            else:
                conv_in_channel = round(hidden_sizes[i-1] * model_ratio)
            conv_out_channel = round(hidden_sizes[i] * model_ratio)

            modules.extend([
                DecomBlock(basis_in_channel, basis_out_channel, conv_in_channel, conv_out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        self.network = nn.Sequential(*modules)

        out_channel = round(hidden_sizes[len(hidden_sizes)-1] * model_ratio)
        self.classifier = nn.Linear(
            int(out_channel * cfg['data_shape'][1] * cfg['data_shape'][2] / (4 ** (len(hidden_sizes)-1))),
            cfg['classes_size']
        )

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
