"""Distribution distance between graph sets: MMD with an RBF kernel on random-GIN embeddings.

Following Thompson et al. (ICLR 2022), graphs are embedded with an untrained GIN (structure only, constant
node features, sum pooling, layer-wise concatenation) and the unbiased MMD^2 is computed between the
embeddings of a reference set and each generated set. The kernel bandwidth is the median pairwise distance
of the reference embeddings.

    python evaluation/mmd_rbf.py --ref data/good_motif-basis-covariate.pkl:train \\
        --gen samples/motif_basis/lam*.pkl --min_conf 0.9
"""
import argparse
import os
import pickle

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


class RandomGIN(nn.Module):
    """Dense GIN with fixed random weights: h <- ReLU(MLP(h + A h)), sum pooling, concatenated over layers."""

    def __init__(self, hidden=35, layers=3):
        super().__init__()
        dims = [1] + [hidden] * layers
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                                  for d in dims[:-1])

    def forward(self, adj, mask):
        h, outs = mask.clone(), []
        for mlp in self.mlps:
            h = torch.relu(mlp(h + torch.bmm(adj, h))) * mask
            outs.append(h.sum(1))
        return torch.cat(outs, 1)


def load_graphs(spec, min_conf=None):
    """`path.pkl` (generated set) or `path.pkl:train|val|test` (a split of a prepared dataset file)."""
    if spec.rsplit(':', 1)[-1] in ('train', 'val', 'test'):
        path, split = spec.rsplit(':', 1)
        with open(path, 'rb') as f:
            return dict(zip(('train', 'val', 'test'), pickle.load(f)))[split]
    with open(spec, 'rb') as f:
        graphs = pickle.load(f)
    conf_path = spec[:-4] + '_confidences.pkl'
    if min_conf is not None and os.path.exists(conf_path):
        with open(conf_path, 'rb') as f:
            graphs = [g for g, c in zip(graphs, pickle.load(f)) if float(c) >= min_conf]
    return graphs


@torch.no_grad()
def embed(model, graphs, batch_size=1024):
    out = []
    for i in range(0, len(graphs), batch_size):
        chunk = [nx.convert_node_labels_to_integers(g) for g in graphs[i:i + batch_size]]
        n_max = max(max(g.number_of_nodes() for g in chunk), 1)
        adj = torch.zeros((len(chunk), n_max, n_max))
        mask = torch.zeros((len(chunk), n_max, 1))
        for j, g in enumerate(chunk):
            mask[j, :g.number_of_nodes(), 0] = 1.0
            if g.number_of_edges():
                e = torch.tensor(list(g.edges()), dtype=torch.long)
                adj[j, e[:, 0], e[:, 1]] = 1.0
                adj[j, e[:, 1], e[:, 0]] = 1.0
        out.append(model(adj, mask))
    return torch.cat(out).numpy().astype(np.float64)


def sq_dist(a, b):
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def kernel_sum(a, b, sigma, block=2000):
    return sum(np.exp(-sq_dist(a[i:i + block], b[j:j + block]) / (2 * sigma ** 2)).sum()
               for i in range(0, len(a), block) for j in range(0, len(b), block))


def mmd2(x, y, sigma):
    """Unbiased estimate of MMD^2 over all graphs of both sets."""
    n, m = len(x), len(y)
    return ((kernel_sum(x, x, sigma) - n) / (n * (n - 1)) + (kernel_sum(y, y, sigma) - m) / (m * (m - 1))
            - 2 * kernel_sum(x, y, sigma) / (n * m))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True, help='reference set, e.g. data/<name>.pkl:train')
    parser.add_argument('--gen', nargs='+', required=True, help='generated sets (.pkl)')
    parser.add_argument('--min_conf', type=float, default=None, help='keep generated graphs with confidence >= this')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model = RandomGIN().eval()
    ref = embed(model, load_graphs(args.ref))
    sub = ref[rng.choice(len(ref), min(2000, len(ref)), replace=False)]
    sigma = float(np.median(np.sqrt(sq_dist(sub, sub))[np.triu_indices(len(sub), 1)]))
    half = len(ref) // 2
    print(f'reference: {args.ref} ({len(ref)} graphs), sigma = {sigma:.3f}')
    print(f'{"reference half vs half (noise floor)":70s} MMD^2 = {mmd2(ref[:half], ref[half:2 * half], sigma):.6f}')
    for path in args.gen:
        if path.endswith(('_labels.pkl', '_confidences.pkl', '.partial.pkl')):    # companions matched by a glob
            continue
        graphs = load_graphs(path, args.min_conf)
        print(f'{os.path.basename(path)[-70:]:70s} MMD^2 = {mmd2(ref, embed(model, graphs), sigma):.6f}  (n = {len(graphs)})',
              flush=True)
