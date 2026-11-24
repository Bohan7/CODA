"""Training of the two learned components of CODA.

ScoreTrainer        unconditional score network s_theta on unlabelled training graphs
                    -> <out_dir>/score.pth (and score_<epoch>.pth every `train.save_interval` epochs)
ClassifierTrainer   time-dependent classifier phi_t on noisy labelled training graphs
                    -> <out_dir>/classifier.pth (the epoch with the best validation accuracy)
"""
import os
import time

import numpy as np
import torch

from losses import get_classifier_loss_fn, get_score_loss_fn
from sde import load_sde
from utils.data import classifier_dataloaders, score_dataloader
from utils.ema import ExponentialMovingAverage
from utils.graph_utils import rand_perm
from utils.loader import (build_model, build_optimizer, classifier_model_params, load_device, load_seed,
                          save_ckpt, score_model_params)


class _Trainer:
    def __init__(self, config, out_dir):
        self.config = config
        self.device = load_device()
        load_seed(config.seed)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.sde_x, self.sde_adj = load_sde(config.sde.x), load_sde(config.sde.adj)

    def _build(self, params):
        self.params = params
        self.model = build_model(params).to(self.device)
        self.optimizer, self.scheduler = build_optimizer(self.model, self.config.train)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.config.train.ema)

    def _step(self, loss):
        cfg = self.config.train
        loss.backward()
        if cfg.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_norm)
        self.optimizer.step()
        self.ema.update(self.model.parameters())

    def _save(self, epoch, name):
        save_ckpt(os.path.join(self.out_dir, f'{name}.pth'), epoch, self.config, self.params, self.model, self.ema)


class ScoreTrainer(_Trainer):
    def train(self):
        cfg = self.config
        loader = score_dataloader(cfg)
        self._build(score_model_params(cfg))
        loss_fn = get_score_loss_fn(self.sde_x, self.sde_adj, cfg.train.lambda_train,
                                    reduce_mean=cfg.train.reduce_mean, eps=cfg.train.eps)

        for epoch in range(cfg.train.num_epochs):
            t_start, loss_x_log, loss_adj_log = time.time(), [], []
            self.model.train()
            for x, adj in loader:
                x, adj = x.to(self.device), adj.to(self.device)
                self.optimizer.zero_grad()
                if cfg.data.perm_mix:
                    x, adj = rand_perm(x, adj)
                loss, loss_x, loss_adj = loss_fn(self.model, x, adj)
                self._step(loss)
                loss_x_log.append(loss_x.item())
                loss_adj_log.append(loss_adj.item())
            if self.scheduler is not None:
                self.scheduler.step()

            print(f'{epoch + 1:05d} | {time.time() - t_start:.2f}s | train x: {np.mean(loss_x_log):.3e} | '
                  f'train adj: {np.mean(loss_adj_log):.3e}', flush=True)
            if epoch == cfg.train.num_epochs - 1:
                self._save(epoch, 'score')
            elif (epoch + 1) % cfg.train.save_interval == 0:
                self._save(epoch, f'score_{epoch + 1}')


class ClassifierTrainer(_Trainer):
    def train(self):
        cfg = self.config
        train_loader, val_loader = classifier_dataloaders(cfg)
        self._build(classifier_model_params(cfg))
        class_weights = None
        if cfg.classifier.get('class_counts'):               # inverse-frequency weights for imbalanced tasks
            class_weights = 1.0 / torch.tensor(cfg.classifier.class_counts).float()
            class_weights = class_weights / class_weights.sum()
        loss_fn = get_classifier_loss_fn(self.sde_x, self.sde_adj, eps=cfg.train.eps, class_weights=class_weights)

        best_val_acc = 0.0
        for epoch in range(cfg.classifier.num_epochs):
            t_start, losses, correct, total = time.time(), [], 0, 0
            self.model.train()
            for x, adj, y in train_loader:
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss, n_correct = loss_fn(self.model, x, adj, y)
                self._step(loss)
                losses.append(loss.item())
                correct, total = correct + n_correct, total + y.size(0)
            if self.scheduler is not None:
                self.scheduler.step()

            self.model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for x, adj, y in val_loader:
                    x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                    _, n_correct = loss_fn(self.model, x, adj, y)
                    val_correct, val_total = val_correct + n_correct, val_total + y.size(0)
            val_acc = val_correct / val_total

            print(f'{epoch + 1:05d} | {time.time() - t_start:.2f}s | train loss: {np.mean(losses):.3e} | '
                  f'train acc: {100 * correct / total:.2f} | val acc: {100 * val_acc:.2f} | '
                  f'best val acc: {100 * best_val_acc:.2f}', flush=True)
            # keep the model with the best accuracy on noisy validation graphs (after a warm-up)
            if val_acc > best_val_acc and epoch >= cfg.classifier.warmup_epochs:
                best_val_acc = val_acc
                self._save(epoch, 'classifier')
