import asyncio
from typing import List

from comm_utils import send_data, get_data
from config import cfg
import copy
import os
import time
from collections import OrderedDict, defaultdict
import random

import numpy as np
import torch
from client import *
import datasets
from models import utils
from training_utils import test

from mpi4py import MPI

import logging

random.seed(cfg['client_selection_seed'])

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['server_cuda']
device = torch.device("cuda" if cfg['server_use_cuda'] and torch.cuda.is_available() else "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

RESULT_PATH = os.getcwd() + '/server_log/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
filename = RESULT_PATH + now + "_" + os.path.basename(__file__).split('.')[0] + '.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

comm_tags = np.ones(cfg['client_num'] + 1)


def main():
    client_num = cfg['client_num']
    logger.info("Total number of clients: {}".format(client_num))
    logger.info("Model type: {}".format(cfg["model_type"]))
    logger.info("Dataset: {}".format(cfg["dataset_type"]))

    # 初始化全局模型
    global_models = OrderedDict()
    for model_ratio in np.unique(cfg['model_ratio']):
        global_models[model_ratio] = utils.create_model_instance(cfg['model_type'], model_ratio)
        global_models[model_ratio].to(device)
        global_params = torch.nn.utils.parameters_to_vector(global_models[model_ratio].parameters())
        para_nums = global_params.nelement()
        model_size = global_params.nelement() * 4 / 1024 / 1024
        logger.info("\nModel ratio: {}".format(model_ratio))
        logger.info("Global params num: {}".format(para_nums))
        logger.info("Global model Size: {} MB".format(model_size))
    global_models = sync_global_models_init(global_models)

    # 划分数据集，客户端有多少个，就把训练集分成多少份
    train_data_partition, partition_sizes = partition_data(
        dataset_type=cfg['dataset_type'],
        partition_pattern=cfg['data_partition_pattern'],
        non_iid_ratio=cfg['non_iid_ratio'],
        client_num=client_num
    )

    logger.info('\nData partition: ')
    for i in range(len(partition_sizes)):
        s = ""
        for j in range(len(partition_sizes[i])):
            s += "{:.3f}".format(partition_sizes[i][j]) + " "
        logger.info(s)

    # create workers
    all_clients: List[ClientConfig] = list()
    for client_idx in range(client_num):
        client = ClientConfig(client_idx)
        client.lr = cfg['lr']
        client.model_ratio = cfg['model_ratio'][client_idx]
        client.train_data_idxes = train_data_partition.use(client_idx)
        all_clients.append(client)

    # 加载测试集
    _, test_dataset = datasets.load_datasets(cfg['dataset_type'], cfg['dataset_path'])
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)

    best_epoch = dict()
    best_acc = dict()
    best_avg_acc = 0
    best_avg_epoch = 1
    # 开始每一轮的训练
    for epoch_idx in range(1, 1 + cfg['epoch_num']):
        logger.info("_____****_____\nEpoch: {:04d}".format(epoch_idx))
        print("_____****_____\nEpoch: {:04d}".format(epoch_idx))

        # 在这里可以实现客户端选择算法，本处实现随机选择客户端id。
        selected_num = cfg['active_client_num']
        model_ratios = np.array(cfg['model_ratio'])
        while True:
            selected_client_idxes = random.sample(range(client_num), selected_num)
            selected_model_ratios = model_ratios[selected_client_idxes]
            flag = True
            for model_ratio in np.unique(model_ratios):
                if len(np.where(selected_model_ratios == model_ratio)[0]) < int(selected_num / len(np.unique(model_ratios))):
                    flag = False
            if flag == True:
                break

        logger.info("Selected clients' idxes: {}".format(selected_client_idxes))
        logger.info("Selected clients' model ratio: {}\n".format(np.array(cfg['model_ratio'])[selected_client_idxes]))
        print("Selected clients' idxes: {}".format(selected_client_idxes))
        print("Selected clients' model ratio: {}\n".format(np.array(cfg['model_ratio'])[selected_client_idxes]))
        selected_clients = []
        for m, client_idx in enumerate(selected_client_idxes):
            all_clients[client_idx].epoch_idx = epoch_idx
            all_clients[client_idx].params_dict = OrderedDict()
            for k, v in global_models[cfg['model_ratio'][client_idx]].state_dict().items():
                all_clients[client_idx].params_dict[k] = copy.deepcopy(v.detach())
            selected_clients.append(all_clients[client_idx])

        # 每一轮都需要将选中的客户端的配置（client.config）发送给相应的客户端
        communication_parallel(selected_clients, action="send_config")

        # 从选中的客户端那里接收配置，此时选中的客户端均已完成本地训练。配置包括训练好的本地模型，学习率等
        communication_parallel(selected_clients, action="get_config")

        # 聚合客户端的本地模型
        global_models = aggregate_model(selected_clients, global_models)


        avg_acc = 0
        avg_loss = 0
        for k, v in global_models.items():
            # 对全局模型进行测试
            test_loss, test_loss_ort, test_acc = test(v, test_loader, device)
            if epoch_idx == 1:
                best_epoch[k] = 1
                best_acc[k] = test_acc
            else:
                if test_acc > best_acc[k]:
                    best_epoch[k] = epoch_idx
                    best_acc[k] = test_acc
            avg_acc += test_acc
            avg_loss += test_loss

            logger.info(
                "Network Fraction: {: .4f}\n".format(k) +
                "Test_Loss: {:.4f}\n".format(test_loss) +
                "Test_Orthogonal_Loss: {:.4f}\n". format(test_loss_ort) +
                "Test_ACC: {:.4f}\n".format(test_acc) +
                "Best_Test_ACC: {:.4f}\n".format(best_acc[k]) +
                "Best_Epoch: {:04d}\n".format(best_epoch[k])
            )

        avg_acc /= len(global_models.items())
        avg_loss /= len(global_models.items())
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_avg_epoch = epoch_idx
            model_save_path = cfg['model_save_path'] + now + '/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path, exist_ok=True)
            for k, v in global_models.items():
                torch.save(v.state_dict(), model_save_path + now + "_" + str(k) + '.pth')

        logger.info(
            "Avg_ACC: {:.4f}\n".format(avg_acc) +
            "Avg_Loss: {:.4f}\n".format(avg_loss) +
            "Best_Avg_ACC: {:.4f}\n".format(best_avg_acc) +
            "Best_Avg_Epoch: {:04d}\n".format(best_avg_epoch)
        )

        for m in range(len(selected_clients)):
            comm_tags[m + 1] += 1


# 参照论文作者代码
# https://github.com/HarukiYqM/All-In-One-Neural-Composition
def aggregate_model(selected_clients, global_models):
    # 聚合basis_conv
    basis_convs = OrderedDict()
    for k, v in list(global_models.values())[0].state_dict().items():
        if 'basis_conv' in k:
            basis_convs[k] = torch.zeros(v.shape, device=device)
    for k in basis_convs:
        for client in selected_clients:
            state_dict = client.params_dict
            basis_convs[k] += copy.deepcopy(state_dict[k].detach()) * (1. / len(selected_clients))
    for model_ratio in global_models:
        global_models[model_ratio].load_state_dict(copy.deepcopy(basis_convs), strict=False)

    # 聚合其他部分的参数，按照不同的网络分簇聚合
    ratio_clients_dict = defaultdict(list)
    for client in selected_clients:
        ratio_clients_dict[client.model_ratio].append(client)
    # 每个网络簇分别聚合
    for model_ratio in global_models:
        model_num = len(ratio_clients_dict[model_ratio])
        client_list = ratio_clients_dict[model_ratio]
        other_convs = OrderedDict()
        for k, v in global_models[model_ratio].state_dict().items():
            if 'basis_conv' not in k and 'num_batches_tracked' not in k:
                other_convs[k] = torch.zeros(v.shape, device=device)
        for k in other_convs:
            for client in client_list:
                other_convs[k] += client.params_dict[k].detach() * (1. / model_num)
        global_models[model_ratio].load_state_dict(copy.deepcopy(other_convs), strict=False)

    return global_models


def sync_global_models_init(global_models):
    basis_convs = {}
    first_global_model = list(global_models.values())[0]
    for k, v in first_global_model.state_dict().items():
        if 'basis_conv' in k:
            basis_convs[k] = v.detach()
    for k in global_models:
        global_models[k].load_state_dict(copy.deepcopy(basis_convs), strict=False)

    return global_models


async def send_config(client, client_rank, comm_tag):
    await send_data(comm, client, client_rank, comm_tag)


async def get_config(client, client_rank, comm_tag):
    config_received = await get_data(comm, client_rank, comm_tag)
    for k, v in config_received.__dict__.items():
        setattr(client, k, v)


def communication_parallel(client_list, action):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for m, client in enumerate(client_list):
        if action == "send_config":
            task = asyncio.ensure_future(send_config(client, m + 1, comm_tags[m+1]))
        elif action == "get_config":
            task = asyncio.ensure_future(get_config(client, m + 1, comm_tags[m+1]))
        else:
            raise ValueError('Not valid action')
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def partition_data(dataset_type, partition_pattern, non_iid_ratio, client_num=10):
    train_dataset, _ = datasets.load_datasets(dataset_type=dataset_type, data_path=cfg['dataset_path'])
    partition_sizes = np.ones((cfg['classes_size'], client_num))
    # iid
    if partition_pattern == 0:
        partition_sizes *= (1.0 / client_num)
    # non-iid
    # 对于每个客户端：包含所有种类的数据，但是某些种类的数据比例非常大
    #
    elif partition_pattern == 1:
        if 0 < non_iid_ratio < 10:
            partition_sizes *= ((1 - non_iid_ratio * 0.1) / (client_num - 1))
            for i in range(cfg['classes_size']):
                partition_sizes[i][i % client_num] = non_iid_ratio * 0.1
        else:
            raise ValueError('Non-IID ratio is too large')
    # non-iid
    # 对于每个客户端：缺少某一部分种类的数据，其余种类的数据按照总体分布分配
    elif partition_pattern == 2:
        if 0 < non_iid_ratio < 10:
            # 计算出每个 worker 缺少多少类数据
            missing_class_num = int(round(cfg['classes_size'] * (non_iid_ratio * 0.1)))

            # 初始化分配矩阵
            partition_sizes = np.ones((cfg['classes_size'], client_num))

            begin_idx = 0
            for worker_idx in range(client_num):
                for i in range(missing_class_num):
                    partition_sizes[(begin_idx + i) % cfg['classes_size']][worker_idx] = 0.
                begin_idx = (begin_idx + missing_class_num) % cfg['classes_size']

            for i in range(cfg['classes_size']):
                count = np.count_nonzero(partition_sizes[i])
                for j in range(client_num):
                    if partition_sizes[i][j] == 1.:
                        partition_sizes[i][j] = 1. / count
        else:
            raise ValueError('Non-IID ratio is too large')
    elif partition_pattern == 3:
        if 0 < non_iid_ratio < 10:
            most_data_proportion = cfg['classes_size'] / client_num * non_iid_ratio * 0.1
            minor_data_proportion = cfg['classes_size'] / client_num * (1 - non_iid_ratio * 0.1) / (cfg['classes_size'] - 1)
            partition_sizes *= minor_data_proportion
            for i in range(client_num):
                partition_sizes[i % cfg['classes_size']][i] = most_data_proportion
        else:
            raise ValueError('Non-IID ratio is too large')
    else:
        raise ValueError('Not valid partition pattern')

    train_data_partition = datasets.LabelwisePartitioner(
        train_dataset, partition_sizes=partition_sizes, seed=cfg['data_partition_seed']
    )

    return train_data_partition, partition_sizes


if __name__ == "__main__":
    main()
