import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout):
        super(GCNAggregator, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, inputs):
        selves, neighbors = inputs
        if self.training:
            selves = F.dropout(selves, self.dropout)
            neighbors = F.dropout(neighbors, self.dropout)

        neighbors = neighbors.view(selves.size(0), -1, selves.size(1))
        selves = selves.unsqueeze(1)

        embeddings = torch.cat((selves, neighbors), dim=1)
        embeddings = embeddings.mean(dim=1)
        embeddings = self.fc(embeddings)
        if self.activation:
            self.activation(embeddings)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class MeanAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout):
        super(MeanAggregator, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.fc = nn.Linear(in_dim*2, out_dim, bias=False)

    def forward(self, inputs):
        selves, neighbors = inputs
        if self.training:
            selves = F.dropout(selves, self.dropout)
            neighbors = F.dropout(neighbors, self.dropout)

        neighbors = neighbors.view(selves.size(0), -1, selves.size(1))
        aggregated_neighbors = neighbors.mean(dim=1)

        embeddings = self.fc(torch.cat((selves, aggregated_neighbors), dim=1))

        if self.activation:
            embeddings = self.activation(embeddings)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class PoolingAggregator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pool_fun, activation, dropout):
        super(PoolingAggregator, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.pool_fun = pool_fun

        self.fc_pool = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc = nn.Linear((in_dim + hidden_dim), out_dim, bias=False)

    def forward(self, inputs):
        selves, neighbors = inputs
        if self.training:
            selves = F.dropout(selves, self.dropout)
            neighbors = F.dropout(neighbors, self.dropout)

        neighbors = self.fc_pool(neighbors)
        neighbors = F.relu(neighbors)
        neighbors = neighbors.view(selves.size(0), -1, neighbors.size(1))
        if self.pool_fun=='max':
            aggregated_neighbors = neighbors.max(dim=1)[0]
        elif self.pool_fun=='mean':
            aggregated_neighbors = neighbors.mean(dim=1)

        embeddings = torch.cat((selves, aggregated_neighbors), dim=1)
        embeddings = self.fc(embeddings)
        if self.activation:
            self.activation(embeddings)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class AttentionAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, heads, activation, dropout, concat=False):  # out_dim = hidden_dim*heads
        super(AttentionAggregator, self).__init__()
        self.heads = heads
        self.activation = activation
        self.dropout = dropout
        self.concat = concat

        hidden_dim = out_dim // heads if self.concat==True else out_dim

        self.fc_x = []
        self.fc_attention=[]
        for i in range(heads):
            self.fc_x.append(nn.Linear(in_dim, hidden_dim, bias=False))
            self.fc_attention.append(nn.Linear(hidden_dim*2, 1, bias=False))
        self.activation = activation

        self.fc_x = nn.Sequential(*self.fc_x)
        self.fc_attention = nn.Sequential(*self.fc_attention)

    def forward(self, inputs):
        selves, neighbors = inputs
        if self.training:
            selves = F.dropout(selves, self.dropout)
            neighbors = F.dropout(neighbors, self.dropout)

        out = None
        for fc_x, fc_attention in zip(self.fc_x.children(), self.fc_attention.children()):
            selves_embd = fc_x(selves)
            neighbors_embd = fc_x(neighbors)
            batch_size = selves.size(0)
            feature_size = selves_embd.size(1)

            selves_repeat = selves_embd.repeat(1, neighbors.size(0) // batch_size + 1)
            selves_repeat = selves_repeat.view(-1, feature_size)

            selves_embd = selves_embd.unsqueeze(1)
            neighbors_embd = neighbors_embd.view(batch_size, -1, feature_size)
            neighbors_embd = torch.cat((neighbors_embd, selves_embd), dim=1)
            neighbors_embd = neighbors_embd.view(-1, feature_size)

            attention_matrix = torch.cat((selves_repeat, neighbors_embd), dim=1)

            attention_coe = fc_attention(attention_matrix)
            attention_coe = F.leaky_relu(attention_coe, 0.2)
            attention_coe = attention_coe.view(batch_size, -1)
            attention_coe = F.softmax(attention_coe, dim=1)
            attention_coe = attention_coe.unsqueeze(1)

            neighbors_embd = neighbors_embd.view(batch_size, -1, feature_size)
            embeddings = torch.matmul(attention_coe, neighbors_embd)
            embeddings = embeddings.squeeze(1)

            if out == None:
                out = embeddings
            else:
                out = torch.cat((out, embeddings), dim=1)

        if self.concat is False:
            out = out.view(selves.size(0), self.heads, -1)
            out = out.mean(dim=1)

        if self.activation:
            out = self.activation(out)

        out = F.normalize(out, p=2, dim=1)
        return out

