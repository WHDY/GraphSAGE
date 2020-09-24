import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Sampler import sampler
from Aggregators import GCNAggregator, MeanAggregator, PoolingAggregator, AttentionAggregator


class layerInfo(object):
    def __init__(self, in_dim, out_dim,
                 hidden_dim, num_samples,
                 aggregator, activation,
                 pool_fun, heads,
                 concat=True):
        super(layerInfo, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.aggregator = aggregator
        self.activation = activation
        self.pool_fun = pool_fun
        self.heads = heads
        self.concat = concat


class supervisedGraphSAGE(nn.Module):
    def __init__(self, layer_info, dropout_rate, neighbors, max_degree, classes):
        super(supervisedGraphSAGE, self).__init__()

        self.sampler = sampler(neighbors, max_degree)

        self.layers = []
        self.num_samples_per_layer = []
        for info in layer_info:
            if info.aggregator == 'GCN':
                self.layers.append(GCNAggregator(info.in_dim, info.out_dim, info.activation, dropout_rate))
            elif info.aggregator == 'Mean':
                self.layers.append(MeanAggregator(info.in_dim, info.out_dim, info.activation, dropout_rate))
            elif info.aggregator == 'Pooling':
                self.layers.append(PoolingAggregator(info.in_dim, info.hidden_dim, info.out_dim,
                                                    info.pool_fun, info.activation, dropout_rate))
            elif info.aggregator == 'Attention':
                self.layers.append(AttentionAggregator(info.in_dim, info.out_dim, info.heads,
                                                       info.activation,  dropout_rate, info.concat))
            elif info.aggregator == 'LSTM':
                pass
            else:
                pass

            self.num_samples_per_layer.append(info.num_samples)

        self.layers = nn.Sequential(*self.layers)
        self.fc = nn.Linear(layer_info[-1].out_dim, classes, bias=True)

    def sample(self, batchNodes):
        whole_batch_nodes = [batchNodes]

        for num_sample in self.num_samples_per_layer:
            sample_neighbors = self.sampler.sample(whole_batch_nodes[-1], num_sample)
            sample_neighbors = sample_neighbors.reshape(sample_neighbors.size(0)*sample_neighbors.size(1))
            whole_batch_nodes.append(sample_neighbors)

        return whole_batch_nodes

    def forward(self, inputs):
        batchNodes, features = inputs
        batchSize = batchNodes.size(0)

        whole_batch_nodes = self.sample(batchNodes)

        hidden_embeddings = [features[nodes] for nodes in whole_batch_nodes]
        next_embeddings = []

        for agg_layer in self.layers.children():
            for i in range(len(hidden_embeddings)-1):
                embeddings = agg_layer((hidden_embeddings[i], hidden_embeddings[i+1]))
                next_embeddings.append(embeddings)
            hidden_embeddings = next_embeddings
            next_embeddings = []

        embeddings = hidden_embeddings[0]

        out = self.fc(embeddings)
        out = F.relu(out)
        out = F.softmax(out, dim=1)
        return out

    def L2_reg(self):
        L2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                L2_reg += torch.square(torch.norm(param))
        return L2_reg
