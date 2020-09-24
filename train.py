import os
from time import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from DataGenerator import dataGenerator
from Models import layerInfo, supervisedGraphSAGE


parser = argparse.ArgumentParser(description="supervisedGraphSAGE")
parser.add_argument('--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

parser.add_argument('--ds', type=str, default='cora', help='which dataset is to test(e.g. cora, pubmed, citeseer)')

parser.add_argument('--num-layers', type=int, default=2, help='number of graph convolutional layers')
parser.add_argument('--agg-type', type=str, default='Mean', help='aggregation method')
parser.add_argument('--pool-fun', type=str, default='max', help='pooling method while using pooling aggregator')
parser.add_argument('--heads', type=int, default=3, help='number of attentions  while using attention aggregator')
parser.add_argument('--h-dim', type=int, default=64, help='hidden dimension while using pooling aggregator')
parser.add_argument('--out-dim', type=str, default='128,128', help='list of hidden dimensions in each layer')
parser.add_argument('--concat', type=str, default='1,0', help='whether to do concat while using AT-agg in each layer')
parser.add_argument('--max-degree', type=int, default=10, help='max degree to make a fix-neighborred matrix')
parser.add_argument('--num-sample', type=str, default='5,5', help='number of sampled neighbors per node in each layer')

parser.add_argument('--bs', type=int, default=100, help='batch size')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--dp', type=float, default=0.05, help='drop rate rate')
parser.add_argument('--wd', type=float, default=5e-4, help='drop rate rate')

parser.add_argument("--val_freq", type=int, default=5, help="model validation frequency(for batch)")
parser.add_argument('--save_freq', type=int, default=5, help='model save frequency(for batch)')
parser.add_argument('--out-file', type=str, default='logdir/visulization', help="file path where store the results")


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def main():
    args = parser.parse_args()
    args = args.__dict__

    # make output file
    # test_mkdir(args['out_file'])
    writer = SummaryWriter(args['out_file'])

    # whether to use gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load data
    dataset = dataGenerator(args['ds'])
    features = torch.tensor(dataset.features).to(device)
    labels = dataset.labels
    classes = dataset.classes
    neighbors = dataset.neighbors

    # make layer information for each layer
    dim = [features.shape[1]]
    out_dim = [int(d) for d in args['out_dim'].strip().split(',')]
    dim.extend(out_dim)

    num_sample = [int(num) for num in args['num_sample'].strip().split(',')]

    concat = [bool(int(cat)) for cat in args['concat'].strip().split(',')]

    layer_infos = []
    for i in range(args['num_layers']):
        layer_infos.append(layerInfo(dim[i], dim[i+1], args['h_dim'],
                                     num_sample[i], args['agg_type'],
                                     lambda x: F.relu(x), args['pool_fun'],
                                     args['heads'], concat[i]))

    # build model
    Net = supervisedGraphSAGE(layer_infos, args['dp'], neighbors, args['max_degree'], classes)
    # writer.add_graph(Net)
    Net.to(device)

    # set optimizer
    optimizer = optim.Adam(Net.parameters(), lr=args['lr'])

    # train
    batchsize = args['bs']
    B = dataset.train_nodes.shape[0] // batchsize
    for epoch in range(args['epochs']):
        print('epoch: {}'.format(epoch+1))
        for i in range(B):
            Net.train()
            batchNodes = dataset.next_batch(batchsize)
            label = labels[batchNodes]

            batchNodes = torch.tensor(batchNodes)
            label = torch.tensor(label).to(device)

            preds = Net((batchNodes, features))

            loss = F.cross_entropy(preds, label)
            # loss = loss + args['wd']*Net.L2_reg()
            writer.add_scalar('training loss', loss, epoch * B + i + 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            preds = torch.argmax(preds, dim=1)
            train_acc = (preds==label).float().mean()
            writer.add_scalar('training accuracy', train_acc, epoch * B + i + 1)
            # print('training accuracy is: {}%'.format(train_acc.item()*100))

            if (epoch*B + i + 1) % args['val_freq'] == 0:
                Net.eval()
                with torch.no_grad():
                    acc_sum = 0.0
                    num = 0
                    for k in range(dataset.test_nodes.shape[0] // 100):
                        start = num*100
                        testBatchNodes = dataset.test_nodes[start: start + 100]
                        test_label = torch.tensor(labels[testBatchNodes]).to(device)
                        testBatchNodes = torch.tensor(testBatchNodes)

                        test_preds = Net((testBatchNodes, features))
                        test_preds = torch.argmax(test_preds, dim=1)
                        acc_sum += (test_preds==test_label).float().mean().item()
                        num += 1
                writer.add_scalar('evaluate accuracy', acc_sum/num, epoch * B + i + 1)
                print('testing accuracy is: {}%'.format(acc_sum*100/num))

            if (epoch*B + i+1) % args['save_freq'] == 0:
                pass

            # make the training curve


if __name__=="__main__":
    main()
