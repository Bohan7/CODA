"""Training data of the generator: padded dense tensors (X, A[, y]) for every dataset.

Generic graphs (GOOD-Motif / GOOD-CMNIST / GOOD-SST2) are read from `<dir>/<name>.pkl`, a tuple of
(train, val, test) lists of networkx graphs written by `data/prepare_good.py`.
Molecules (GOOD-HIV) are read from `<dir>/<name>_train_kekulized.npz`, written by `data/prepare_good_hiv.py`.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.graph_utils import graph_node_flags, graphs_to_tensor, mask_x, node_flags

MOL_DATASETS = ('GoodHiv', 'GoodHivSize')
# atom vocabulary of the 10-atom-type subset the molecule generators are trained on (0 = padding)
ATOM_LIST = {'GoodHiv': [6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 0],
             'GoodHivSize': [6, 7, 8, 9, 11, 15, 16, 17, 35, 53, 0]}


def is_mol(config):
    return config.data.data in MOL_DATASETS


# ---------------------------------------------------------------- generic graphs
def load_graph_lists(config):
    """(train, val) lists of networkx graphs.

    The OOD validation graphs of the size and length splits are larger than `max_node_num`. For these splits
    the first 3,000 training graphs serve as the validation set; they remain part of the training set.
    """
    name = config.data.data
    with open(os.path.join(config.data.dir, f'{name}.pkl'), 'rb') as f:
        train, val, _ = pickle.load(f)
    if 'size' in name or 'sst2' in name:
        val = train[:3000]
    print(f'Dataset sizes: train {len(train)}, val {len(val)}')
    return train, val


def graphs_to_xa(config, graphs):
    """Dense node features and adjacency of a list of graphs.

    `data.node_features` selects the node representation:
      degree        one-hot node degree (GOOD-Motif)
      attr_degree   node attribute `x` concatenated with the one-hot degree (GOOD-CMNIST: RGB + degree)
      attr          node attribute `x` only (GOOD-SST2: 768-d word embeddings)
    """
    kind, max_n, max_f = config.data.node_features, config.data.max_node_num, config.data.max_feat_num
    if kind == 'degree':
        adj = graphs_to_tensor(graphs, max_n)
        attr = None
    else:
        adj, attr = graphs_to_tensor(graphs, max_n, node_attr='x')
    flags = node_flags(adj)
    if kind == 'attr':
        return mask_x(attr, flags), adj
    num_deg = max_f if attr is None else max_f - attr.shape[-1]
    deg = mask_x(F.one_hot(adj.sum(dim=-1).to(torch.long), num_classes=num_deg).to(torch.float32), flags)
    x = deg if attr is None else torch.cat([attr, deg], dim=-1)
    return x, adj


def graph_labels(graphs):
    return torch.tensor([int(g.graph['label']) for g in graphs], dtype=torch.long)


# ---------------------------------------------------------------- molecules
def load_mol_npz(path):
    """List of (atomic numbers, bond channels, label) tuples."""
    if not os.path.exists(path):
        raise ValueError(f'Invalid filepath {path} for dataset')
    data = np.load(path)
    arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    return list(zip(*arrays))


def filter_mols(mols, dataset):
    """Keep the molecules whose atoms all belong to the 10-atom-type vocabulary."""
    atoms = np.array(ATOM_LIST[dataset])
    return [m for m in mols if np.all(np.isin(np.array(m[0]), atoms))]


def mols_to_xa(mols, dataset):
    """One-hot atom types (padding column dropped) and bond-order matrices with entries in {0, 1, 2, 3}."""
    atom_list = ATOM_LIST[dataset]
    xs, adjs = [], []
    for mol in mols:
        atoms, bonds = mol[0], mol[1]
        x = np.zeros((len(atoms), len(atom_list)), dtype=np.float32)
        for i, a in enumerate(atoms):
            x[i, atom_list.index(a)] = 1.
        # channels: single, double, triple, no bond
        bonds = np.concatenate([bonds[:3], 1 - np.sum(bonds[:3], axis=0, keepdims=True)], axis=0)
        order = torch.tensor(bonds.astype(np.float32).argmax(axis=0))
        xs.append(torch.tensor(x)[:, :-1])
        adjs.append(torch.where(order == 3, 0, order + 1).to(torch.float32))
    return torch.stack(xs), torch.stack(adjs)


def load_mols(config, split):
    path = os.path.join(config.data.dir, f'{config.data.data.lower()}_{split}_kekulized.npz')
    return filter_mols(load_mol_npz(path), config.data.data)


# ---------------------------------------------------------------- public API
def train_node_flags(config):
    """Node masks of the training graphs: the empirical size prior used at sampling time."""
    if is_mol(config):
        _, adj = mols_to_xa(load_mols(config, 'train'), config.data.data)
        return node_flags(adj)
    train, _ = load_graph_lists(config)
    return graph_node_flags(train, config.data.max_node_num)


def score_dataloader(config):
    """Unlabelled (X, A) loader for training the score network."""
    if is_mol(config):
        x, adj = mols_to_xa(load_mols(config, 'train'), config.data.data)
    else:
        train, _ = load_graph_lists(config)
        x, adj = graphs_to_xa(config, train)
    return DataLoader(TensorDataset(x, adj), batch_size=config.data.batch_size, shuffle=True)


def classifier_dataloaders(config):
    """Labelled (X, A, y) train / validation loaders for training the time-dependent classifier."""
    if is_mol(config):
        mols = load_mol_npz(os.path.join(config.data.dir, f'{config.data.data.lower()}_train_kekulized.npz'))
        pos = [m for m in mols if m[2] == 1]
        neg = [m for m in mols if m[2] == 0]
        n_neg, n_pos = config.classifier.val_per_class          # fixed class-stratified validation split
        splits = {'train': pos[n_pos:] + neg[n_neg:], 'val': pos[:n_pos] + neg[:n_neg]}
        tensors = {}
        for name, split in splits.items():
            split = filter_mols(split, config.data.data)
            x, adj = mols_to_xa(split, config.data.data)
            y = torch.tensor(np.array([m[2] for m in split]), dtype=torch.long).view(-1)
            tensors[name] = (x, adj, y)
    else:
        train, val = load_graph_lists(config)
        tensors = {name: (*graphs_to_xa(config, graphs), graph_labels(graphs))
                   for name, graphs in (('train', train), ('val', val))}
    return tuple(DataLoader(TensorDataset(*tensors[name]), batch_size=config.data.batch_size, shuffle=True)
                 for name in ('train', 'val'))
