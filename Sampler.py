import numpy as np
import torch
from DataGenerator import dataGenerator


class sampler(object):
    def __init__(self, neighbors, max_degree):
        super(sampler, self).__init__()
        self.neighbors = neighbors
        self.max_degree = max_degree
        self.fix_neighbors = self.random_fixed_neighbors()

    def sample(self, nodes, num):
        neighbors = self.fix_neighbors[nodes]

        order = np.arange(neighbors.size(1))
        np.random.shuffle(order)
        # shuffle = torch.tensor(order)

        shuffle_neighbors = (neighbors.t())[order]
        neighbors = shuffle_neighbors.t()

        return neighbors[0:, 0: num]

    def random_fixed_neighbors(self):
        fixed_neighbors = np.zeros(shape=(len(self.neighbors), self.max_degree), dtype=np.longlong)
        for node in self.neighbors:
            if len(self.neighbors[node]) < self.max_degree:
                fixed_neighbors[node, :] = np.random.choice(list(self.neighbors[node]), self.max_degree, replace=True)
            else:
                fixed_neighbors[node, :] = np.random.choice(list(self.neighbors[node]), self.max_degree, replace=False)

        return torch.tensor(fixed_neighbors)

    def re_fixed_neighbors(self):
        self.fix_neighbors = self.random_fixed_neighbors()


if __name__=="__main__":
    cora = dataGenerator('cora')
    features = cora.features
    label = cora.labels
    neighbors = cora.neighbors


    max_degree = 5

    nodes = torch.tensor([0, 1, 4, 2, 100])
    num_samples = 3

    sampler = sampler(neighbors, max_degree)
    sampled_neighbor = sampler.sample(nodes, num_samples)
    print(sampled_neighbor)
    print(sampler.sample(nodes, 3))
    print(sampler.sample(nodes, 3))
    sampler.re_fixed_neighbors()
    next_nodes = sampler.sample(nodes, 3)
    next_nodes = next_nodes.reshape(next_nodes.size(0)*next_nodes.size(1))
    print(sampler.sample(next_nodes, 3))

