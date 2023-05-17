import copy
import os
import random
import time
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tools.dataset import Dataset
from tools.common import Accumulator
from network.MGN import MGN
from config.config import load_train_config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def accumulate(model, dataloader, config):
    node_accumulator = Accumulator(config["network"]["node_feat_size"])
    edge_accumulator = Accumulator(config["network"]["edge_feat_size"])
    # output_accumulator = Accumulator(config["network"]["output_feat_size"])
    for i, (_, _, nodes, edges, output, path) in enumerate(dataloader):
        nodes = nodes.cuda()
        edges = edges.cuda()
        output = output.cuda()

        node_accumulator.accumulate(nodes)
        edge_accumulator.accumulate(edges)
        # output_accumulator.accumulate(output)

    model.node_normalizer.set_accumulated(node_accumulator)
    model.edge_normalizer.set_accumulated(edge_accumulator)
    # network.output_normalizer.set_accumulated(output_accumulator)


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config):
    log = open(os.path.join(config['log_root'], 'log.txt'), 'a')
    result = 0.
    accumulate(model, train_dataloader, config)
    for epoch in range(1, config['max_epoch'] + 1):

        print('-' * 20)
        # log.write('-' * 20 + '\n')

        print('Epoch: %d / %d' % (epoch, config['max_epoch']))
        # log.write('Epoch: %d / %d' % (epoch, config['max_epoch']) + '\n')

        epoch_loss = 0.
        epoch_mse = 0.
        model.train()
        start = time.perf_counter()
        for i, (senders, receivers, nodes, edges, label, _) in enumerate(tqdm(train_dataloader)):
            # data scale,cuda()不改变数据大小
            # torch.Size([1, 106973])
            # torch.Size([1, 106973])
            # torch.Size([1, 15287, 3])
            # torch.Size([1, 106973, 4])
            # torch.Size([1])
            senders = senders.cuda()
            receivers = receivers.cuda()
            nodes = nodes.cuda()
            edges = edges.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            prediction = model(senders, receivers, nodes, edges)
            # print(prediction.dtype, label.dtype)
            loss = criterion(prediction, label)
            loss.backward()
            epoch_loss += loss.item()
            # epoch_mse += torch.mean((label - prediction) ** 2)

            optimizer.step()
            scheduler.step()

        end = time.perf_counter()
        print('Train Loss: %f' % (epoch_loss / len(train_dataloader)))
        print('Train Time: %f' % (end - start))
        """
        log.write('Train Loss: %f' % (epoch_loss / len(train_dataloader)) + '\n')
"""
        # if epoch % config['eval_steps'] == 0:
        #     test(network, valid_dataloader,criterion)
        if epoch % config['max_epoch'] == 0:
            epoch_loss = 0.
            epoch_mse = 0.
            epoch_time = 0.

            model.eval()
            for i, (senders, receivers, nodes, edges, label, _) in enumerate(valid_dataloader):
                senders = senders.cuda()
                receivers = receivers.cuda()
                nodes = nodes.cuda()
                edges = edges.cuda()
                label = label.cuda()

                with torch.no_grad():
                    start = time.perf_counter()
                    prediction = model(senders, receivers, nodes, edges)
                    end = time.perf_counter()

                    # loss = criterion(prediction, network.output_normalizer(label))
                    loss = criterion(prediction, label)
                    epoch_loss += loss.item()
                    # epoch_mse += torch.mean((label - network.output_normalize_inverse(prediction)) ** 2)
                    # epoch_mse += torch.mean((label - prediction) ** 2)
                    epoch_time += end - start

            print('Valid Loss: %f, Time Used: %f' % (
                epoch_loss / len(valid_dataloader), epoch_time / len(valid_dataloader)))
            result = epoch_loss / len(valid_dataloader)
            # if config is not None:
            #     log = open(os.path.join(config['log_root'], 'log.txt'), 'a')
            #     log.write('Valid Loss: %f' % (epoch_loss / len(valid_dataloader)) + '\n')
        """
        print('-' * 20)
        log.write('-' * 20 + '\n')

        if epoch % config['save_steps'] == 0:
            torch.save(copy.deepcopy((network.state_dict())), os.path.join(config['ckpt_root'], '%d.pkl' % epoch))
        """
    return result


def main(message_passing_step, learning_rate, gamma):
    config = load_train_config()
    random.seed(config['seed'])

    model = MGN(config['network'], message_passing_step)
    model.cuda()

    train_dataloader = data.DataLoader(Dataset(config['dataset'], 5, 'train'), batch_size=config['batch_size'],
                                       num_workers=1,
                                       pin_memory=True)
    valid_dataloader = data.DataLoader(Dataset(config['dataset'], 5, 'val'), batch_size=config['batch_size'],
                                       num_workers=1,
                                       pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    loss = train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config)
    return loss


if __name__ == '__main__':
    main(8,0.0001,0.9)
