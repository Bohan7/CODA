"""Post-hoc generate-then-filter baselines GDSS-CCD and GDSS-CCM.

Class-conditional GDSS samples (lambda = 0) are ranked by an OOD score computed on graph embeddings and only
the `keep_ratio` fraction with the highest score is kept (over the whole set, or separately for every class):
  density       negative log-likelihood under a 5-component Gaussian mixture fitted on training embeddings
  mahalanobis   Mahalanobis distance to the training embeddings
"""
import numpy as np
import torch
from torch_geometric.loader import DataLoader


class OODFilter:
    def __init__(self, method='density'):
        assert method in ('density', 'mahalanobis')
        self.method = method

    @staticmethod
    def _embed(batch, model, device):
        with torch.no_grad():
            return model.feat_encoder(data=batch.to(device)).cpu()

    def fit(self, train_loader, model, device):
        model.eval()
        feats = torch.cat([self._embed(batch, model, device) for batch in train_loader], dim=0)
        if self.method == 'density':
            from sklearn.mixture import GaussianMixture
            self.gmm = GaussianMixture(n_components=5, random_state=42)
            self.gmm.fit(feats.numpy())
        else:
            self.mean = feats.mean(dim=0)
            cov = torch.cov(feats.T)
            self.cov_inv = torch.linalg.pinv(cov + 1e-5 * torch.eye(cov.shape[0]))

    def score(self, batch, model, device):
        feat = self._embed(batch, model, device)
        if self.method == 'density':
            return -self.gmm.score_samples(feat.numpy())
        diff = feat - self.mean
        return torch.sqrt(torch.sum(diff @ self.cov_inv * diff, dim=1)).numpy()

    def filter(self, graphs, model, device, keep_ratio=0.5, per_class=False):
        model.eval()
        scores = []
        for batch in DataLoader(graphs, batch_size=64, shuffle=False, drop_last=False):
            scores.extend(self.score(batch, model, device))
        scores = np.array(scores)

        groups = {}                                   # one group, or one group per class
        for i, g in enumerate(graphs):
            groups.setdefault(g.y.item() if per_class else 'all', []).append(i)
        kept = []
        for name, indices in groups.items():
            threshold = np.percentile(scores[indices], (1 - keep_ratio) * 100)
            kept_group = [i for i in indices if scores[i] > threshold]
            kept.extend(kept_group)
            if per_class:
                print(f'  class {name}: {len(indices)} -> {len(kept_group)} graphs')
        print(f'OOD filter ({self.method}): {len(graphs)} -> {len(kept)} graphs (keep_ratio={keep_ratio})')
        return [graphs[i] for i in kept]
