"""Dense graph utilities: masking, noise, quantisation and graph <-> tensor conversion."""
import networkx as nx
import numpy as np
import torch


# ---------------------------------------------------------------- masks and noise
def mask_x(x, flags):
    """Zero the features of padded nodes. x: B x N x F, flags: B x N."""
    return x * flags[:, :, None]


def mask_adjs(adjs, flags):
    """Zero the rows/columns of padded nodes. adjs: B x N x N or B x C x N x N, flags: B x N."""
    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


def node_flags(adj, eps=1e-5):
    """A node is present iff it has at least one incident edge."""
    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)
    if len(flags.shape) == 3:
        flags = flags[:, 0, :]
    return flags


def gen_noise(x, flags, sym=True):
    """Masked Gaussian noise; symmetric with zero diagonal for adjacency matrices."""
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1, -2)
        return mask_adjs(z, flags)
    return mask_x(z, flags)


def rand_perm(x, adj):
    """Random node permutation of every graph in the batch (training-time augmentation)."""
    flags = node_flags(adj)
    batch_size, num_nodes = adj.shape[0], adj.shape[-1]
    idx = torch.arange(num_nodes, device=adj.device).repeat(batch_size, 1)
    for b in range(batch_size):
        pidx = torch.where(flags[b] > 0)[0]
        idx[b][pidx] = idx[b][pidx[torch.randperm(int(flags[b].sum(-1)))]]
    peye = torch.eye(num_nodes, device=adj.device)[idx]
    px = torch.bmm(peye, x)
    padj = torch.bmm(torch.bmm(peye, adj), peye.transpose(-1, -2))
    return px, padj


def pow_tensor(x, cnum):
    """Stack the first `cnum` powers of the adjacency matrix: B x N x N -> B x cnum x N x N."""
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum - 1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    return torch.cat(xc, dim=1)


# ---------------------------------------------------------------- initial node masks for sampling
def graph_node_flags(graph_list, max_node_num):
    """Node mask (N_graphs x max_node_num) of every training graph, used as the empirical size prior."""
    flags = torch.zeros((len(graph_list), max_node_num), dtype=torch.float32)
    for i, g in enumerate(graph_list):
        adj = nx.to_numpy_array(g, nodelist=list(g.nodes))
        if adj.shape[0] > max_node_num:
            raise ValueError(f'graph with {adj.shape[0]} nodes exceeds max_node_num={max_node_num}')
        present = np.abs(adj).sum(-1) > 1e-5
        flags[i, :adj.shape[0]] = torch.from_numpy(present.astype(np.float32))
    return flags


def init_flags(train_flags, batch_size):
    """Sample node masks from the empirical node-count distribution of the training graphs."""
    idx = np.random.randint(0, train_flags.shape[0], batch_size)
    return train_flags[idx]


def init_random_flags(min_node_num, max_node_num, batch_size):
    """Node-count prior for size exploration: n ~ Uniform{min_node_num, ..., max_node_num}.

    As in the paper's implementation the mask is built from a path graph on n nodes, so n = 1 yields an
    empty mask (a single node has no incident edge).
    """
    num_nodes = np.random.randint(min_node_num, max_node_num + 1, batch_size)
    flags = torch.zeros((batch_size, max_node_num), dtype=torch.float32)
    for i, n in enumerate(num_nodes):
        if n >= 2:
            flags[i, :n] = 1.
    return flags


# ---------------------------------------------------------------- quantisation
def quantize(adjs, thr=0.5):
    return torch.where(adjs < thr, torch.zeros_like(adjs), torch.ones_like(adjs))


def quantize_mol(adjs):
    """Round a continuous bond matrix to bond orders {0, 1, 2, 3}."""
    adjs = adjs.detach().cpu()
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return np.array(adjs.to(torch.int64))


# ---------------------------------------------------------------- graphs <-> tensors
def pad_adjs(adj, node_number):
    n = adj.shape[-1]
    if n > node_number:
        raise ValueError(f'graph with {n} nodes exceeds max_node_num={node_number}')
    out = np.zeros((node_number, node_number), dtype=adj.dtype)
    out[:n, :n] = adj
    return out


def graphs_to_tensor(graph_list, max_node_num, node_attr=None):
    """Padded adjacency tensor of a list of networkx graphs.

    If `node_attr` is given, also return the padded node-attribute tensor (N_graphs x max_node_num x F).
    """
    adjs, feats = [], []
    for g in graph_list:
        nodes = list(g.nodes)
        adjs.append(pad_adjs(nx.to_numpy_array(g, nodelist=nodes), max_node_num))
        if node_attr is not None:
            f = np.asarray([g.nodes[v][node_attr] for v in nodes], dtype=np.float32)
            padded = np.zeros((max_node_num, f.shape[-1]), dtype=np.float32)
            padded[:f.shape[0]] = f
            feats.append(padded)
    adjs = torch.tensor(np.asarray(adjs), dtype=torch.float32)
    if node_attr is None:
        return adjs
    return adjs, torch.tensor(np.asarray(feats), dtype=torch.float32)


def adjs_to_graphs(adjs, labels, xs=None):
    """Convert sampled (quantised) adjacency matrices into labelled networkx graphs.

    Isolated nodes are removed. When `xs` is given, every node keeps its generated feature vector under
    the node attribute `features`.
    """
    graphs = []
    for i, adj in enumerate(adjs):
        g = nx.from_numpy_array(adj.detach().cpu().numpy())
        g.remove_edges_from(nx.selfloop_edges(g))
        if xs is not None:
            for v, feat in enumerate(xs[i].detach().cpu().numpy()):
                g.nodes[v]['features'] = feat
        g.remove_nodes_from(list(nx.isolates(g)))
        if g.number_of_nodes() < 1:
            g.add_node(1)
        g.graph['label'] = labels[i].item()
        graphs.append(g)
    return graphs
