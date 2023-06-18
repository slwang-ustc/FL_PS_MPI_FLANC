import sys
sys.path.append('/data/slwang/FL_PS_MPI_FLANC')
import random
import logging
from torch.utils.data import DataLoader
from torch import optim
import time
import os
import math
from cnn_HeteroFL import create_cnn

import torch
from torch import nn

from config import cfg
import datasets


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes * BasicBlock.expansion, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * BasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes * BasicBlock.expansion)
            )

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        out = residual + shortcut
        out = nn.ReLU(inplace=True)(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, num_block):
        super(ResNet, self).__init__()

        # torch.manual_seed(cfg['model_init_seed'])

        data_shape = cfg['data_shape']
        classes_size = cfg['classes_size']

        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(data_shape[0], 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, 64, num_block[0], 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, classes_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avg_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output


def resnet18():
    """
    return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def train(device, model_dir, logger):
    total_iters = 25000
    freq = 50
    batch_size = 32
    test_batch_size = 32
    lr = 0.01
    best_acc = 0.0
    best_epoch = 1
    model_ratio = 1.15625

    # load the testing and training dataset
    train_dataset, test_dataset = datasets.load_datasets(cfg['dataset_type'], cfg['dataset_path'])

    # load the testing and training loader of the client
    train_loader = datasets.DataLoaderHelper(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4))
    test_loader = datasets.DataLoaderHelper(DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=4))

    # model = resnet18_flanc(model_ratio)
    # model = resnet18()
    # model = CNN_FLANC(model_ratio)
    model = create_cnn(model_ratio)
    model_params = torch.nn.utils.parameters_to_vector(model.parameters())
    para_nums = model_params.nelement()
    model_size = para_nums * 4 / 1024 / 1024
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        momentum=0.9,
        lr=lr,
        weight_decay=5e-4
    )

    expLr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    loss_func = nn.CrossEntropyLoss()

    logger.info(
        "Total_iters: {:04d}\n".format(total_iters) +
        "Local_update_frequency: {}\n".format(freq) +
        "Batch_size: {}\n".format(batch_size) +
        "Model_type: CNN_HeteroFL\n" +
        "Dataset: {}\n".format(cfg['dataset_type']) +
        "Model_ratio: {}".format(model_ratio)
    )
    logger.info("Model params num: {}".format(para_nums))
    logger.info("Model Size: {} MB\n".format(model_size))

    t_start  = time.time()
    for iter in range(1, total_iters + 1):
        model.train()
        data, target = next(train_loader)
            
        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        train_loss = loss_func(output, target)
        # loss = F.nll_loss(output, target)
        train_loss.backward()
        optimizer.step()
        if iter % 50 == 0 and iter > 200:
            expLr.step()


        pred = output.argmax(1, keepdim=True)
        correct_num = pred.eq(target.view_as(pred)).sum().item()
        train_acc = correct_num / len(data)

        if iter % freq == 0:
            t_end = time.time()
            model.eval()
            data_loader = test_loader.loader

            test_loss = 0.0
            correct = 0

            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    # sum up batch loss
                    test_loss += loss_func(output, target).item() * data.shape[0]
                    # get the index of the max log-probability
                    pred = output.argmax(1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_acc = correct / len(data_loader.dataset)
            test_loss /= len(data_loader.dataset)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = int(iter / freq)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), model_dir + now + "_" + os.path.basename(__file__).split('.')[0] + '.pth')

            logger.info("_____****_____\nEpoch: {:04d}    lr: {:.4f}".format(int(iter / freq), optimizer.param_groups[0]['lr']))

            logger.info(
                "Train_ACC: {:.4f}\n".format(train_acc) +
                "Train_Loss: {:.4f}\n".format(train_loss.item()) +
                "Train_Time: {:.4f}\n".format(t_end - t_start)
            )

            logger.info(
                "Test_ACC: {:.4f}\n".format(test_acc) +
                "Test_Loss: {:.4f}\n".format(test_loss) +
                "Best_epoch: {:04d}\n".format(best_epoch) +
                "Best_ACC: {:.4f}\n".format(best_acc)
            )
            t_start = time.time()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda")

random.seed(2020)
torch.manual_seed(2020)

log_path = '/data/slwang/results_temp/logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path, exist_ok=True)
# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
filename = log_path + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
model_dir = '/data/slwang/results_temp/models/'

train(device, model_dir, logger)