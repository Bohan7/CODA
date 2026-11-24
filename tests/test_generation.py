"""Unit tests of the generation stage (CPU only, no data or checkpoints required)."""
import glob
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'generation')
sys.path.insert(0, ROOT)

from models.transformer import GraphTransformer                                      # noqa: E402
from sde import VESDE, VPSDE                                                         # noqa: E402
from solver import get_coda_sampler, get_coda_score_fn, get_score_fn                 # noqa: E402
from utils.config import load_config                                                 # noqa: E402
from utils.graph_utils import init_random_flags, mask_adjs, mask_x, node_flags       # noqa: E402

HIDDEN = {'dx': 16, 'de': 8, 'dy': 8, 'n_head': 2, 'dim_ffX': 16, 'dim_ffE': 8}
MLP = {'X': 16, 'E': 8, 'y': 16}


def tiny_models(feat=4, classes=3):
    torch.manual_seed(0)
    score = GraphTransformer(2, {'X': feat, 'E': 2, 'y': 1}, MLP, HIDDEN, {'X': feat, 'E': 1, 'y': 0})
    clf = GraphTransformer(1, {'X': feat, 'E': 2, 'y': 1}, MLP, HIDDEN, {'X': feat, 'E': 1, 'y': classes})
    return score.eval(), clf.eval()


def noisy_batch(batch=3, n=6, feat=4):
    torch.manual_seed(1)
    flags = torch.ones(batch, n)
    flags[0, 4:] = 0
    x = mask_x(torch.randn(batch, n, feat), flags)
    adj = torch.randn(batch, n, n).triu(1)
    adj = mask_adjs(adj + adj.transpose(-1, -2), flags)
    return x, adj, flags


@pytest.mark.parametrize('sde_adj', [VPSDE(0.1, 1.0, 10), VESDE(0.2, 1.0, 10)])
def test_coda_score_is_tempered_guided_score(sde_adj):
    """score(lambda) = (1 - sqrt(lambda)) * score(0), and lambda = 0 without guidance is s_theta."""
    score, clf = tiny_models()
    sde_x = VPSDE(0.1, 1.0, 10)
    x, adj, flags = noisy_batch()
    t = torch.full((x.shape[0],), 0.5)
    labels = torch.tensor([0, 1, 2])
    base = get_score_fn(sde_x, sde_adj, score)

    plain = base(x, adj, flags, t)
    unguided = get_coda_score_fn(base, clf, labels, 0.0, 0.0, lam=0.0)(x, adj, flags, t)
    assert all(torch.equal(a, b) for a, b in zip(plain, unguided))

    guided = get_coda_score_fn(base, clf, labels, 0.5, 0.2, lam=0.0)(x, adj, flags, t)
    tempered = get_coda_score_fn(base, clf, labels, 0.5, 0.2, lam=0.09)(x, adj, flags, t)
    for g, s in zip(guided, tempered):
        assert torch.allclose(s, (1 - math.sqrt(0.09)) * g, atol=1e-6)
    assert not torch.allclose(guided[0], plain[0])          # the classifier gradient changes the node score


@pytest.mark.parametrize('predictor,corrector', [('Euler', 'Langevin'), ('Reverse', 'None'), ('Euler', 'None')])
def test_sampler_shapes_masks_and_determinism(predictor, corrector):
    score, clf = tiny_models()
    sde_x, sde_adj = VPSDE(0.1, 1.0, 5), VPSDE(0.1, 1.0, 5)
    flags = torch.ones(3, 6)
    flags[1, 3:] = 0
    labels = torch.tensor([0, 1, 2])
    sampler = get_coda_sampler(sde_x, sde_adj, (3, 6, 4), (3, 6, 6), predictor=predictor, corrector=corrector,
                               snr=0.2, scale_eps=0.7, weight_x=0.5, lam=0.04, eps=1e-3, device='cpu')
    torch.manual_seed(0)
    x1, adj1 = sampler(score, clf, flags, labels)
    torch.manual_seed(0)
    x2, adj2 = sampler(score, clf, flags, labels)
    assert x1.shape == (3, 6, 4) and adj1.shape == (3, 6, 6)
    assert torch.equal(x1, x2) and torch.equal(adj1, adj2)
    assert torch.all(x1[1, 3:] == 0) and torch.all(adj1[1, 3:] == 0) and torch.all(adj1[1, :, 3:] == 0)
    assert torch.allclose(adj1, adj1.transpose(-1, -2), atol=1e-5)
    assert all(p.grad is None for p in clf.parameters())      # guidance needs input gradients only


def test_unknown_solver_names_are_rejected():
    with pytest.raises(ValueError):
        get_coda_sampler(VPSDE(0.1, 1.0, 5), VPSDE(0.1, 1.0, 5), (1, 4, 4), (1, 4, 4), predictor='euler')
    with pytest.raises(ValueError):
        get_coda_sampler(VPSDE(0.1, 1.0, 5), VPSDE(0.1, 1.0, 5), (1, 4, 4), (1, 4, 4), corrector='langevin')


def test_sde_marginals_and_reverse():
    x = torch.randn(2, 4, 3)
    t = torch.tensor([0.3, 0.9])
    for sde in (VPSDE(0.1, 1.0, 100), VESDE(0.2, 1.0, 100)):
        mean, std = sde.marginal_prob(x, t)
        assert mean.shape == x.shape and std.shape == (2,) and torch.all(std > 0)
        drift, diffusion = sde.reverse().sde(x, t, torch.zeros_like(x))
        fwd_drift, fwd_diffusion = sde.sde(x, t)
        assert torch.equal(drift, fwd_drift) and torch.equal(diffusion, fwd_diffusion)   # zero score -> forward drift


def test_node_count_prior():
    np.random.seed(0)
    flags = init_random_flags(min_node_num=2, max_node_num=9, batch_size=500)
    counts = flags.sum(1)
    assert flags.shape == (500, 9) and counts.min() >= 2 and counts.max() == 9
    assert torch.equal(flags, (torch.arange(9)[None] < counts[:, None]).float())           # prefix masks


def test_node_flags_from_adjacency():
    adj = torch.zeros(1, 4, 4)
    adj[0, 0, 1] = adj[0, 1, 0] = 1
    assert node_flags(adj).tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_configs_load_and_override():
    paths = sorted(glob.glob(os.path.join(ROOT, 'configs', '*.yaml')))
    assert len(paths) == 6
    for path in paths:
        cfg = load_config(path, ['sampler.ood=0.03', 'sample.n_graphs=12', 'ckpt.score=some/path.pth'])
        assert cfg.sampler.ood == 0.03 and cfg.sample.n_graphs == 12 and cfg.ckpt.score == 'some/path.pth'
        assert 0 <= cfg.sampler.weight_x_classifier <= 1
        assert len(cfg.sample.class_ratio) == cfg.classifier.num_classes
        with pytest.raises(KeyError):                         # a misspelt entry must not be ignored
            load_config(path, ['sampler.lambda=0.03'])
        assert load_config(path, ['sample.use_ema=false']).sample.use_ema is False
        assert load_config(path, ['sampler.corrector=Langevin']).sampler.corrector == 'Langevin'
