import sys
sys.path.append('/data/slwang/FL_PS_MPI_FLANC/models')
import torch
import torch.nn as nn
from cnn_flanc import CNN_FLANC
from resnet_flanc import resnet18_flanc


def create_model_instance(model_type, model_ratio):
    if model_type == 'cnn_flanc':
        return CNN_FLANC(model_ratio)
    if model_type == 'resnet18_flanc':
        return resnet18_flanc(model_ratio)
    else:
        raise ValueError("Not valid model type")


def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss_fun


def orth_loss(model, model_type, para_loss_type='L2', device=torch.device('cpu')):
    """ Orthogonal regularizer proposed by the Flanc paper  """
    loss_fun = loss_type(para_loss_type)
    loss = 0
    if model_type == 'cnn_flanc':
        for i in range(len(model.network)):
            if i % 3 == 1:
                basis_conv_paras = model.network[i].basis_conv.weight
                b = basis_conv_paras.view(basis_conv_paras.shape[0], -1)
                d = torch.mm(b, torch.t(b))
                d = loss_fun(d, torch.eye(basis_conv_paras.shape[0], basis_conv_paras.shape[0], device=device))
                loss = loss + d
    elif model_type == 'resnet18_flanc':
        for i in range(len(model.network)):
            for j in range(len(model.network[i])):
                for m in [0, 3]:
                    basis_conv_paras = model.network[i][j].residual_function[m].basis_conv.weight
                    b = basis_conv_paras.view(basis_conv_paras.shape[0], -1)
                    d = torch.mm(b, torch.t(b))
                    d = loss_fun(d, torch.eye(basis_conv_paras.shape[0], basis_conv_paras.shape[0], device=device))
                    loss = loss + d
    return loss
