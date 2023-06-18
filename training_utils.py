import time
import math

import torch
import torch.nn as nn

from config import cfg
from models.utils import orth_loss


def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu")):
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)

    train_loss_ort = 0.0
    train_loss = 0.0
    samples_num = 0

    t_start = time.time()
    loss_func = nn.CrossEntropyLoss()
    for iter_idx in range(local_iters):
        data, target = next(data_loader)
            
        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        loss = loss_func(output, target)
        loss_orth = cfg['lambda_loss'] * orth_loss(model, cfg['model_type'], 'L2', device)
        loss += loss_orth
        loss.backward()
        optimizer.step()

        train_loss_ort += (loss_orth.item() * data.size(0))
        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss_ort /= samples_num
        train_loss /= samples_num

    t_end = time.time()
    
    return train_loss_ort, train_loss, t_end-t_start


def test(model, data_loader, device=torch.device("cpu")):
    model.to(device)
    model.eval()
    data_loader = data_loader.loader

    test_loss_ort = 0.0
    test_loss = 0.0
    correct = 0

    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_func(output, target).item() * data.shape[0]
            test_loss_ort += cfg['lambda_loss'] * orth_loss(model, cfg['model_type'], 'L2', device).item() * data.shape[0]
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = (test_loss + test_loss_ort) / len(data_loader.dataset)
    test_loss_ort = test_loss_ort / len(data_loader.dataset)
    test_accuracy = correct / len(data_loader.dataset)

    # TODO: Record

    return test_loss, test_loss_ort, test_accuracy
