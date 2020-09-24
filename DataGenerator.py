import numpy as np
from collections import defaultdict


class dataGenerator(object):
    def __init__(self, datasetName):
        self.datasetName = datasetName
        self.nodes = None
        self.train_nodes = None
        self.test_nodes = None

        self.features = None
        self.labels = None
        self.neighbors = None

        boudary_line = None
        if self.datasetName == "cora":
            self.features, self.labels, self.neighbors, self.classes = load_cora()
            boudary_line = 1708
        else:
            pass

        # normalize features
        features_l2_norm = np.linalg.norm(self.features, axis=1, keepdims=True)
        self.features = self.features / features_l2_norm

        # split training nodes and testing nodes
        self.nodes = np.arange(self.features.shape[0], dtype=np.longlong)
        self.train_nodes = np.arange(boudary_line, dtype=np.longlong)
        self.test_nodes = np.arange(boudary_line, self.features.shape[0], dtype=np.longlong)

        self.batch_index_in_epoch = 0
        np.random.shuffle(self.train_nodes)

    def next_batch(self, batchSize):
        start = self.batch_index_in_epoch
        self.batch_index_in_epoch += batchSize
        if self.batch_index_in_epoch > len(self.train_nodes):
            np.random.shuffle(self.train_nodes)
            start = 0
            self.batch_index_in_epoch = batchSize
        end = self.batch_index_in_epoch
        return self.train_nodes[start: end]


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats), dtype=np.float32)
    labels = np.empty((num_nodes), dtype=np.int64)

    node_map = {}  # convert node id to corresponding features index
    label_map = {}  # convert class name to integer

    with open("data/cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()  # info: <paper_id>, <features>, <class_label>
            feat_data[i, :] = [float(feature) for feature in info[1: -1]]

            node_map[info[0]] = i

            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("data/cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()  # <cited paper id> <---- <cite paper id>

            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]

            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    return feat_data, labels, adj_lists, len(label_map)


if __name__=="__main__":
    '''
    #  test cora data set
    features, labels, adj_list = load_cora()
    print(type(features), features.shape, features)
    print(type(labels), labels.shape, labels)
    print(type(adj_list), len(adj_list))

    total_degree = 0
    max_degree = 0
    for node in adj_list:
        total_degree += len(adj_list[node])
        if max_degree < len(adj_list[node]):
            max_degree = len(adj_list[node])

    print(total_degree / features.shape[0])
    print(max_degree)
    '''
    cora = dataGenerator('cora')
    features = cora.features
    labels = cora.labels

    iter = 0
    neighbors = cora.neighbors
    for node in neighbors:
        print(type(node))
        print(type(neighbors[node]), neighbors[node])
        iter+=1
        if iter >= 10:
            break

    print(type(features), features.shape, features)
