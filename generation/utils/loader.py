"""Seeding, devices, model construction and checkpoint I/O."""
import random

import numpy as np
import torch
from easydict import EasyDict as edict

from models.transformer import GraphTransformer
from utils.config import to_plain
from utils.ema import ExponentialMovingAverage


def load_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def score_model_params(config):
    """Constructor arguments of the score network s_theta."""
    return {'n_layers': config.model.num_layers,
            'input_dims': {'X': config.data.max_feat_num, 'E': config.model.input_dims.E, 'y': 1},   # y = time
            'hidden_mlp_dims': dict(config.model.hidden_mlp_dims),
            'hidden_dims': dict(config.model.hidden_dims),
            'output_dims': {'X': config.data.max_feat_num, 'E': 1, 'y': 0}}


def classifier_model_params(config):
    """Constructor arguments of the time-dependent classifier phi_t (same backbone, class logits on `y`)."""
    params = score_model_params(config)
    params['n_layers'] = config.classifier.num_layers
    params['output_dims']['y'] = config.classifier.num_classes
    return params


def build_model(params):
    return GraphTransformer(**params)


def build_optimizer(model, cfg_train):
    if cfg_train.optimizer != 'adamw':
        raise NotImplementedError(f'Optimizer {cfg_train.optimizer} is not implemented.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr, amsgrad=True,
                                  weight_decay=cfg_train.weight_decay)
    scheduler = None
    if cfg_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg_train.lr_decay)
    return optimizer, scheduler


def save_ckpt(path, epoch, config, params, model, ema):
    """Checkpoints hold only tensors and plain python containers, so they load with `weights_only=True`."""
    torch.save({'epoch': epoch, 'config': to_plain(config), 'params': to_plain(params),
                'state_dict': model.state_dict(), 'ema': ema.state_dict()}, path)


def load_ckpt(path, device):
    """Load a checkpoint written by `save_ckpt`."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    ckpt['config'] = edict(ckpt['config'])
    print(f'{path} loaded')
    return ckpt


def load_model_from_ckpt(ckpt, device, use_ema=False):
    model = build_model(ckpt['params'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    if use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=ckpt['ema']['decay'])
        ema.load_state_dict(ckpt['ema'])
        ema.copy_to(model.parameters())
    return model
