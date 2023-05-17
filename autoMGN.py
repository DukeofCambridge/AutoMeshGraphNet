import argparse
import logging
import nni
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from nni.utils import merge_parameter

from tools.dataset import Dataset
from tools.common import Accumulator
from network.MGN import MGN
from tqdm import tqdm
from config.config import load_train_config
logger = logging.getLogger('AutoGNN')


def accumulate(model, dataloader, config):
    node_accumulator = Accumulator(config["model"]["node_feat_size"])
    edge_accumulator = Accumulator(config["model"]["edge_feat_size"])
    # output_accumulator = Accumulator(config["model"]["output_feat_size"])
    for i, (_, _, nodes, edges, output, path) in enumerate(dataloader):
        nodes = nodes.cuda()
        edges = edges.cuda()
        output = output.cuda()

        node_accumulator.accumulate(nodes)
        edge_accumulator.accumulate(edges)
        # output_accumulator.accumulate(output)

    model.node_normalizer.set_accumulated(node_accumulator)
    model.edge_normalizer.set_accumulated(edge_accumulator)
    # model.output_normalizer.set_accumulated(output_accumulator)


def train(args, model, train_loader, optimizer, scheduler, config, epoch, device):
    accumulate(model, train_loader, config)
    model.train()
    for batch_idx, (senders, receivers, nodes, edges, label, _) in enumerate(tqdm(train_loader)):
        senders = senders.to(device)
        receivers = receivers.to(device)
        nodes = nodes.to(device)
        edges = edges.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(senders, receivers, nodes, edges)
        criterion = nn.MSELoss()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args['batch_size'], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (senders, receivers, nodes, edges, label, _) in test_loader:
            senders = senders.to(device)
            receivers = receivers.to(device)
            nodes = nodes.to(device)
            edges = edges.to(device)
            label = label.to(device)
            output = model(senders, receivers, nodes, edges)
            # sum up batch loss
            criterion = nn.MSELoss()
            test_loss += criterion(output, label).item()

            # get the index of the max log-probability
            # pred = output.argmax(dim=1, keepdim=True)
            # correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))

    return test_loss


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    config = load_train_config()
    random.seed(args['seed'])
    dataset_addr = 'data/CarModel'+'_S'+str(args['mesh_size'])

    train_dataloader = data.DataLoader(Dataset(dataset_addr, 5, 'train'), batch_size=args['batch_size'],
                                       num_workers=1,
                                       pin_memory=True,shuffle=True)
    valid_dataloader = data.DataLoader(Dataset(dataset_addr, 5, 'val'), batch_size=args['batch_size'],
                                       num_workers=1,
                                       pin_memory=True,shuffle=True)
    model = MGN(config['model'], args).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args['gamma'])

    result = 0.0
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, train_dataloader, optimizer, scheduler, config, epoch, device)
        test_acc = test(args, model, valid_dataloader, device)

        # report intermediate result
        nni.report_intermediate_result(test_acc)
        logger.debug('test loss %f', test_acc)
        logger.debug('Pipe send intermediate result done.')
        result = test_acc
    # report final result
    nni.report_final_result(result)
    logger.debug('Final result is %g', result)
    logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='AutoGNN')
    parser.add_argument("--data_dir", type=str,
                        default='./data/CarModel', help="data directory")
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--message_passing_step', type=int, default=15,
                        help='how many times to update the feature of nodes and edges')
    parser.add_argument('--latent_size', type=int, default=128,
                        help="hidden dimension of basic MLP")
    parser.add_argument('--num_layers', type=int, default=4,
                        help='layer num of basic MLP')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='decayRate of scheduler of learning_rate')
    parser.add_argument('--agg_fun', type=str, default='sum',
                        help='how to combine the features of neighbors and self')
    parser.add_argument('--activation', type=str, default='elu',
                        help='activation function of latent mlp')
    parser.add_argument('--mesh_size', type=int, default=50)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        # params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise