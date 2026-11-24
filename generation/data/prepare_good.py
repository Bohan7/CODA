"""Convert a GOOD covariate-shift split into the generator's training file `<out_dir>/<name>.pkl`.

    python data/prepare_good.py --dataset motif  --domain basis  --good_dir ../data/GOOD
    python data/prepare_good.py --dataset motif  --domain size   --good_dir ../data/GOOD
    python data/prepare_good.py --dataset cmnist --domain color  --good_dir ../data/GOOD
    python data/prepare_good.py --dataset sst2   --domain length --good_dir ../data/GOOD

The file holds (train, val, test) lists of networkx graphs with `graph['label']` (and the node attribute
`x` for GOOD-CMNIST / GOOD-SST2). GOOD downloads the dataset into `--good_dir` on first use.
"""
import argparse
import os
import pickle
import sys

import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'classification'))


def load(dataset, good_dir, domain):
    if dataset == 'motif':
        from GOOD.data.good_datasets.good_motif import GOODMotif as cls
    elif dataset == 'cmnist':
        from GOOD.data.good_datasets.good_cmnist import GOODCMNIST as cls
    else:
        from GOOD.data.good_datasets.good_sst2 import GOODSST2 as cls
    return cls.load(good_dir, domain=domain, shift='covariate', generate=False)[0]


def to_graphs(split, with_node_attr):
    graphs = []
    for data in DataLoader(split, batch_size=1, shuffle=False):
        g = to_networkx(data, node_attrs=['x'] if with_node_attr else None, to_undirected=True)
        assert isinstance(g, nx.Graph)
        g.graph['label'] = data.y.item()
        if hasattr(data, 'env_id'):
            g.graph['env_id'] = data.env_id.item()
        graphs.append(g)
    return graphs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['motif', 'cmnist', 'sst2'])
    parser.add_argument('--domain', required=True, help='basis | size (motif), color (cmnist), length (sst2)')
    parser.add_argument('--good_dir', default='../data/GOOD')
    parser.add_argument('--out_dir', default='data')
    args = parser.parse_args()

    dataset = load(args.dataset, args.good_dir, args.domain)
    splits = tuple(to_graphs(dataset[s], with_node_attr=args.dataset != 'motif') for s in ('train', 'val', 'test'))
    print('graphs per split (train / val / test):', [len(s) for s in splits])
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'good_{args.dataset}-{args.domain}-covariate.pkl')
    with open(path, 'wb') as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved', path)
