"""Unit tests of the downstream stage (CPU only, no GOOD data required)."""
import os
import pickle
import sys

import networkx as nx
import numpy as np
import pytest
import torch
from munch import Munch
from torch_geometric.data import Batch, Data

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'classification'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from data import DATASETS, load_generated_graphs, load_generated_molecules      # noqa: E402
from ood_filter import OODFilter                                                # noqa: E402
from significance_test import paired_bootstrap                                  # noqa: E402
from train import build_model, evaluate                                         # noqa: E402


def write_set(tmp_path, graphs, labels, confidences, name='set.pkl'):
    path = str(tmp_path / name)
    for suffix, obj in (('.pkl', graphs), ('_labels.pkl', labels), ('_confidences.pkl', confidences)):
        with open(path[:-4] + suffix, 'wb') as f:
            pickle.dump(obj, f)
    return path


def test_load_generated_graphs_filters_and_connects(tmp_path):
    def from_edges(n, edges):                               # generated graphs carry unit edge weights
        adj = np.zeros((n, n))
        for i, j in edges:
            adj[i, j] = adj[j, i] = 1.0
        return nx.from_numpy_array(adj)

    connected = from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    split = from_edges(4, [(0, 1), (2, 3)])                 # two components -> joined by one edge
    low_conf = from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    edgeless = nx.Graph()
    edgeless.add_node(1)                                    # what the sampler stores for a graph without edges
    path = write_set(tmp_path, [connected, split, low_conf, edgeless], [0, 2, 1, 1], [0.95, 0.99, 0.5, 0.99])
    graphs = load_generated_graphs(path, 'motif_basis', min_confidence=0.9)
    assert len(graphs) == 2                                 # low confidence and edgeless graphs are skipped
    assert [g.y.item() for g in graphs] == [0, 2]
    assert all(g.env_id.item() == 1 and g.x.shape == (g.num_nodes, 1) for g in graphs)
    assert graphs[1].edge_index.shape[1] == 2 * 3           # 2 original edges + 1 connecting edge, both directions


def test_load_generated_graphs_with_features(tmp_path):
    g = nx.from_numpy_array(np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]))
    for v in g.nodes:
        g.nodes[v]['features'] = np.arange(44, dtype=np.float32) + v
    path = write_set(tmp_path, [g], [7], [1.0])
    data = load_generated_graphs(path, 'cmnist_color', min_confidence=0.9)[0]
    assert data.x.shape == (3, 3) and data.y.item() == 7    # RGB channels only
    data = load_generated_graphs(path, 'sst2_length', min_confidence=0.9)[0]
    assert data.x.shape == (3, 44) and data.y.shape == (1, 1) and data.y.dtype == torch.float


def test_load_generated_molecules_ranks_by_confidence(tmp_path):
    path = str(tmp_path / 'mols.txt')
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'not_a_smiles']
    with open(path, 'w') as f:
        f.write('\n'.join(smiles) + '\n')
    for suffix, obj in (('_labels.pkl', [0, 0, 1, 0, 0]), ('_confidences.pkl', [0.91, 0.99, 0.95, 0.5, 0.99])):
        with open(path[:-4] + suffix, 'wb') as f:
            pickle.dump(obj, f)
    mols = load_generated_molecules(path, min_confidence=0.9, max_per_class=(1, 5))
    assert [m.smiles for m in mols] == ['c1ccccc1', 'CC(=O)O']            # top-1 inactive, then the active one
    assert all(m.env_id.item() == 1 and m.y.shape == (1, 1) for m in mols)
    assert len(load_generated_molecules(path, min_confidence=None, max_per_class=(5, 5))) == 4


@pytest.mark.parametrize('name,backbone', [('motif_basis', 'GIN'), ('motif_basis', 'GCN'), ('motif_basis', 'GAT'),
                                           ('cmnist_color', 'vGIN'), ('hiv_scaffold', 'vGIN'), ('hiv_scaffold', 'GCN')])
def test_backbones_forward_and_evaluate(name, backbone):
    torch.manual_seed(0)
    spec = DATASETS[name]
    args = Munch(backbone=backbone, dim_hidden=32, dropout_rate=0.5, model_layer=3, device='cpu')
    model = build_model(args, spec, meta_info=None)
    graphs = []
    for i in range(8):
        n = 5 + i % 3
        edge_index = torch.tensor([[j, (j + 1) % n] for j in range(n)] + [[(j + 1) % n, j] for j in range(n)]).t()
        if spec['type'] == 'mol':
            g = Data(x=torch.zeros(n, 9, dtype=torch.long), edge_index=edge_index,
                     edge_attr=torch.zeros(2 * n, 3, dtype=torch.long), y=torch.tensor([[i % 2]]))
        else:
            g = Data(x=torch.rand(n, spec['dim_node']), edge_index=edge_index, y=torch.tensor([i % spec['num_classes']]))
        graphs.append(g)
    out = model(data=Batch.from_data_list(graphs))
    assert out.shape == (8, spec['num_classes'])
    from torch_geometric.loader import DataLoader
    score = evaluate(model, DataLoader(graphs, batch_size=4), 'cpu', spec['task'])
    assert 0 <= score <= 100


class _Encoder(torch.nn.Module):
    """Stand-in for a GNN: the graph embedding is the mean node feature."""

    def feat_encoder(self, data):
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(data.x, data.batch)


@pytest.mark.parametrize('method', ['density', 'mahalanobis'])
def test_ood_filter_keeps_far_graphs(method):
    from torch_geometric.loader import DataLoader
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    train = [Data(x=torch.randn(2, 4) * 0.1, edge_index=edge_index, y=torch.tensor([0])) for _ in range(200)]
    near = [Data(x=torch.randn(2, 4) * 0.1, edge_index=edge_index, y=torch.tensor([0])) for _ in range(20)]
    far = [Data(x=torch.randn(2, 4) * 0.1 + 5.0, edge_index=edge_index, y=torch.tensor([0])) for _ in range(20)]
    flt = OODFilter(method)
    flt.fit(DataLoader(train, batch_size=32), _Encoder(), 'cpu')
    kept = flt.filter(near + far, _Encoder(), 'cpu', keep_ratio=0.5)
    assert len(kept) == 20 and all(g.x.mean() > 2 for g in kept)
    # per class: half of every class is kept, whatever its distance
    for g in far:
        g.y = torch.tensor([1])
    kept = flt.filter(near + far, _Encoder(), 'cpu', keep_ratio=0.5, per_class=True)
    assert sorted(g.y.item() for g in kept) == [0] * 10 + [1] * 10


def test_paired_bootstrap():
    a = np.array([75.0, 76.0, 74.5, 77.0, 75.5])
    delta, p = paired_bootstrap(a, a - 2.0)
    assert delta == pytest.approx(2.0) and p == pytest.approx(1 / 20001)
    delta, p = paired_bootstrap(a, a[::-1].copy())
    assert abs(delta) < 1e-9 and 0.3 < p <= 1.0
