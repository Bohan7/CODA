"""Datasets of the downstream OOD classification stage and loading of CODA augmentation sets."""
import pickle
import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from GOOD.utils.data import from_smiles

# Per-dataset settings of the main results in the paper (every value can be overridden on the command line).
# `milestones` are the epochs at which the learning rate is multiplied by 0.1; `var_weight` is the weight of the
# variance penalty between the mean loss on original and on generated graphs. `aug` describes how a generated
# set is used:
#   min_confidence   keep generated graphs whose classifier confidence p(y | G) reaches this value
#                    (graphs: >=, molecules: >; None = no confidence filter)
#   max_per_class    (molecules) keep at most this many most-confident molecules of class 0 / class 1
DATASETS = {
    'motif_basis': dict(good='GOODMotif', domain='basis', type='syn', num_classes=3, dim_node=1, task='multiclass',
                        backbone='GIN', model_layer=5, epochs=100, milestones=[80], lr=1e-3, batch_size=32,
                        test_epoch=80, var_weight=0.5, optimizer='adamw', seed=676, aug=dict(min_confidence=0.9)),
    'motif_size': dict(good='GOODMotif', domain='size', type='syn', num_classes=3, dim_node=1, task='multiclass',
                       backbone='GIN', model_layer=5, epochs=100, milestones=[], lr=1e-3, batch_size=32,
                       test_epoch=0, var_weight=1.0, optimizer='adamw', seed=676, aug=dict(min_confidence=0.9)),
    'cmnist_color': dict(good='GOODCMNIST', domain='color', type='syn', num_classes=10, dim_node=3, task='multiclass',
                         backbone='vGIN', model_layer=5, epochs=100, milestones=[], lr=1e-3, batch_size=128,
                         test_epoch=0, var_weight=0.0, optimizer='adamw', seed=676, aug=dict(min_confidence=0.9)),
    'sst2_length': dict(good='GOODSST2', domain='length', type='real', task='binary',
                        backbone='vGIN', model_layer=3, epochs=200, milestones=[150], lr=1e-3, batch_size=32,
                        test_epoch=0, var_weight=0.0, optimizer='adamw', seed=676, aug=dict(min_confidence=0.9)),
    'hiv_scaffold': dict(good='GOODHIV', domain='scaffold', type='mol', num_classes=1, task='binary_auc',
                         backbone='vGIN', model_layer=3, epochs=200, milestones=[150], lr=1e-4, batch_size=32,
                         test_epoch=0, var_weight=0.0, optimizer='adam', seed=133,
                         aug=dict(min_confidence=0.9, max_per_class=(12000, 500))),
    'hiv_size': dict(good='GOODHIV', domain='size', type='mol', num_classes=1, task='binary_auc',
                     backbone='vGIN', model_layer=3, epochs=200, milestones=[150], lr=1e-3, batch_size=32,
                     test_epoch=0, var_weight=0.0, optimizer='adam', seed=133,
                     aug=dict(min_confidence=None, max_per_class=(5000, 200))),
}


def load_good_dataset(name, data_dir):
    """GOOD dataset under covariate shift: dict with `train`, `id_val`, `val` (OOD), `test` (OOD), ..."""
    spec = DATASETS[name]
    if spec['good'] == 'GOODMotif':
        from GOOD.data.good_datasets.good_motif import GOODMotif as cls
    elif spec['good'] == 'GOODCMNIST':
        from GOOD.data.good_datasets.good_cmnist import GOODCMNIST as cls
    elif spec['good'] == 'GOODSST2':
        from GOOD.data.good_datasets.good_sst2 import GOODSST2 as cls
    else:
        from GOOD.data.good_datasets.good_hiv import GOODHIV as cls
    return cls.load(data_dir, domain=spec['domain'], shift='covariate', generate=False)


# ---------------------------------------------------------------- augmentation sets
def _read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def connect_graph(g):
    """Join the connected components of a generated graph with one random edge between consecutive components."""
    components = list(nx.connected_components(g))
    for i in range(len(components) - 1):
        node_a = random.choice(list(components[i]))
        node_b = random.choice(list(components[i + 1]))
        g.add_edge(node_a, node_b, weight=1.0)
        g.add_edge(node_b, node_a, weight=1.0)
    assert nx.is_connected(g)
    return g


def load_generated_graphs(path, name, min_confidence):
    """Generated networkx graphs (`<path>.pkl` with `_labels.pkl` / `_confidences.pkl`) as PyG `Data` objects.

    Graphs below the confidence threshold and graphs without edges are skipped. Every graph gets `env_id = 1`
    (the original training graphs are environment 0).
    """
    spec = DATASETS[name]
    labels = _read_pickle(path[:-4] + '_labels.pkl')
    confidences = _read_pickle(path[:-4] + '_confidences.pkl')
    binary = spec['task'] == 'binary'
    graphs = []
    for j, g in enumerate(_read_pickle(path)):
        if confidences[j] < min_confidence or g.number_of_edges() == 0:
            continue
        if not nx.is_connected(g):
            g = connect_graph(g)
        data = from_networkx(g)
        if name.startswith('motif'):                    # GOOD-Motif has constant node features
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
        else:
            feats = np.array([g.nodes[v]['features'] for v in g.nodes])
            if name.startswith('cmnist'):               # generated features = [RGB | one-hot degree]; keep RGB
                feats = feats[:, :3]
            data.x = torch.tensor(feats, dtype=torch.float)
        data.y = torch.tensor([[labels[j]]], dtype=torch.float) if binary else torch.tensor([labels[j]], dtype=torch.long)
        data.env_id = torch.tensor([1], dtype=torch.long)
        for key in ('weight', 'features', 'label'):     # attributes collected by from_networkx
            if hasattr(data, key):
                delattr(data, key)
        graphs.append(data)
    return graphs


def load_generated_molecules(path, min_confidence, max_per_class):
    """Generated SMILES (`<path>.txt` with `_labels.pkl` / `_confidences.pkl`) as PyG `Data` objects.

    Molecules are ranked by classifier confidence; the `max_per_class` most confident molecules of the
    inactive / active class are kept.
    """
    labels = _read_pickle(path[:-4] + '_labels.pkl')
    confidences = _read_pickle(path[:-4] + '_confidences.pkl')
    with open(path, 'r') as f:
        smiles = [line.strip() for line in f]

    mols = list(zip(confidences, labels, smiles))
    if min_confidence is not None:
        mols = [m for m in mols if m[0] > min_confidence]
    mols.sort(key=lambda m: m[0], reverse=True)

    per_class = {}
    for confidence, label, smile in mols:
        try:
            data, _ = from_smiles(smile)
        except Exception:                               # SMILES outside the featurizer's vocabulary
            continue
        if data.x is None or data.x.size(0) == 0:       # unparsable SMILES
            continue
        data.y = torch.tensor([[label]], dtype=torch.long)
        data.env_id = torch.tensor([1], dtype=torch.long)
        per_class.setdefault(label, []).append(data)
    return per_class.get(0, [])[:max_per_class[0]] + per_class.get(1, [])[:max_per_class[1]]


def load_augmentation(path, name):
    spec = DATASETS[name]
    if spec['type'] == 'mol':
        return load_generated_molecules(path, spec['aug']['min_confidence'], spec['aug']['max_per_class'])
    return load_generated_graphs(path, name, spec['aug']['min_confidence'])


# ---------------------------------------------------------------- original training graphs
def original_graphs(train_set, name):
    """The original training graphs as individual `Data` objects (environment 0).

    The molecule datasets are read in shuffled order, as in the code used for the paper; this keeps the
    random-number stream, and therefore every seed's result, identical to that code.
    """
    mol = DATASETS[name]['type'] == 'mol'
    graphs = []
    for batch in DataLoader(train_set, batch_size=1, shuffle=mol, num_workers=0):
        common = dict(x=batch.x, edge_index=batch.edge_index, env_id=torch.zeros_like(batch.env_id), y=batch.y)
        if mol:
            graphs.append(Data(edge_attr=batch.edge_attr, smiles=batch.smiles, **common))
        else:
            graphs.append(Data(num_nodes=batch.num_nodes, **common))
    return graphs
