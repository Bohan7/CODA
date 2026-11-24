"""CODA downstream stage: train a GNN on the original training graphs plus a generated augmentation set.

    python train.py --dataset motif_basis --aug ../generation/samples/motif_basis/coda.pkl
    python train.py --dataset hiv_scaffold --aug ../generation/samples/hiv_scaffold/coda.txt --trials 10
    python train.py --dataset cmnist_color                       # ERM, no augmentation

Every epoch the model is evaluated on the ID-validation, OOD-validation and OOD-test splits. The reported
test score is the one at the best epoch of the selection split (`--select val`: OOD validation, the GOOD
protocol; `--select id_val`: ID validation). The score under the other protocol is logged as well.
Accuracy is reported for GOOD-Motif / GOOD-CMNIST / GOOD-SST2 and ROC-AUC for GOOD-HIV, both in percent.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from munch import Munch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.loader import DataLoader

from data import DATASETS, load_augmentation, load_good_dataset, original_graphs
from ood_filter import OODFilter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args, spec, meta_info):
    """GNN backbone from the GOOD benchmark (GIN, GIN with virtual node, GCN or GAT)."""
    if args.backbone == 'GIN':
        from GOOD.networks.models.GINs import GIN as cls
    elif args.backbone == 'vGIN':
        from GOOD.networks.models.GINvirtualnode import vGIN as cls
    elif args.backbone == 'GCN':
        from GOOD.networks.models.GCNs import GCN as cls
    else:
        from GOOD.networks.models.GAT import GAT as cls

    if spec['type'] == 'mol':
        dataset = Munch(dataset_type='mol', num_classes=1, dim_edge=3)
    elif spec['type'] == 'real':
        dataset = Munch(dataset_type='real', num_classes=meta_info.num_classes, dim_node=meta_info.dim_node,
                        dim_edge=meta_info.dim_edge)
    else:
        dataset = Munch(dataset_type='syn', num_classes=spec['num_classes'], dim_node=spec['dim_node'], dim_edge=0)
    config = Munch(model=Munch(dim_hidden=args.dim_hidden, dim_ffn=args.dim_hidden, dropout_rate=args.dropout_rate,
                               model_layer=args.model_layer, global_pool='mean', model_level='graph'),
                   dataset=dataset, device=args.device)
    return cls(config)


@torch.no_grad()
def evaluate(model, loader, device, task):
    """Accuracy in percent (multiclass / binary) or ROC-AUC in [0, 1] (binary_auc)."""
    model.eval()
    if task == 'binary_auc':
        y_true, y_pred = [], []
        for batch in loader:
            batch = batch.to(device)
            if batch.x.shape[0] == 1:               # a batch that consists of a single node is skipped,
                continue                            # as in the evaluation loop of the code used for the paper
            pred = model(data=batch)
            y_true.append(batch.y.view(pred.shape).cpu())
            y_pred.append(pred.cpu())
        return roc_auc_score(torch.cat(y_true).numpy()[:, 0], torch.cat(y_pred).numpy()[:, 0])
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(data=batch)
        if task == 'binary':
            correct += (pred.view(-1) > 0).long().eq(batch.y.view(-1).long()).sum().item()
        else:
            correct += pred.max(1)[1].eq(batch.y.view(-1)).sum().item()
    return correct / len(loader.dataset) * 100


def build_training_set(args, spec, dataset, meta_info, device):
    """Augmentation graphs (environment 1) followed by the original training graphs (environment 0)."""
    graphs = []
    if args.aug:
        print(f'Loading augmentation set: {args.aug}', flush=True)
        graphs = load_augmentation(args.aug, args.dataset)
        if args.ood_filter:                         # GDSS-CCD / GDSS-CCM baselines
            encoder = build_model(args, spec, meta_info).to(device)
            if args.pretrain_ckpt:
                encoder.load_state_dict(torch.load(args.pretrain_ckpt, map_location=device))
            else:
                print('OOD filter: no --pretrain_ckpt given, embeddings come from a randomly initialised encoder')
            ood_filter = OODFilter(args.ood_filter)
            ood_filter.fit(DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, drop_last=True),
                           encoder, device)
            graphs = ood_filter.filter(graphs, encoder, device, keep_ratio=args.ood_filter_keep_ratio,
                                       per_class=args.ood_filter_per_class)
    num_aug = len(graphs)
    graphs += original_graphs(dataset['train'], args.dataset)
    print(f'Training graphs: {len(graphs) - num_aug} original + {num_aug} generated', flush=True)
    return graphs, num_aug


def run(args, seed):
    spec = DATASETS[args.dataset]
    set_seed(seed)
    if spec['type'] == 'mol':
        torch.backends.cudnn.enabled = False
    device = torch.device(args.device)
    task = spec['task']

    dataset, meta_info = load_good_dataset(args.dataset, args.data_dir)
    other = 'id_val' if args.select == 'val' else 'val'
    loaders = {split: DataLoader(dataset[name], batch_size=args.batch_size, shuffle=False, num_workers=0)
               for split, name in ((args.select, args.select), ('test', args.test_split), (other, other))}

    graphs, num_aug = build_training_set(args, spec, dataset, meta_info, device)
    # a final batch of a single graph cannot be batch-normalised: drop it
    train_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(graphs) % args.batch_size == 1)
    save_ckpt = args.save_ckpt
    if save_ckpt and args.trials > 1:               # one file per seed
        root, ext = os.path.splitext(save_ckpt)
        save_ckpt = f'{root}_seed{seed}{ext}'

    model = build_model(args, spec, meta_info).to(device)
    if spec['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    if task == 'multiclass':
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    best = {args.select: 0, other: 0}              # best validation score per selection split
    test_at = {args.select: 0, other: 0}           # test score at that epoch
    for epoch in range(1, args.epochs + 1):
        t_start, total_loss = time.time(), 0
        for batch in train_loader:
            batch = batch.to(device)
            model.train()
            optimizer.zero_grad()
            pred = model(data=batch)
            if task == 'multiclass':
                losses = criterion(pred, batch.y)
            elif task == 'binary':
                losses = criterion(pred.view(-1), batch.y.view(-1))
            else:
                losses = criterion(pred.to(torch.float32), batch.y.to(torch.float32))
            loss = losses.mean()
            if args.var_weight > 0:                 # optional V-REx-style penalty across the two environments
                env_losses = [losses[batch.env_id == e].mean() for e in (0, 1) if (batch.env_id == e).any()]
                penalty = args.var_weight * torch.var(torch.stack(env_losses))
                if not torch.isnan(penalty):
                    loss = penalty + loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scores = {split: evaluate(model, loaders[split], device, task) for split in (args.select, 'test', other)}
        for split in (other, args.select):
            if scores[split] > best[split] and epoch > args.test_epoch:
                best[split], test_at[split] = scores[split], scores['test']
                if split == args.select and save_ckpt:
                    os.makedirs(os.path.dirname(save_ckpt) or '.', exist_ok=True)
                    torch.save(model.state_dict(), save_ckpt)
        scheduler.step()
        print(f'Epoch [{epoch}/{args.epochs}] loss {total_loss / len(train_loader):.4f} | '
              f'{args.select} {scores[args.select]:.4f} test {scores["test"]:.4f} {other} {scores[other]:.4f} | '
              f'test@best-{args.select} {test_at[args.select]:.4f} test@best-{other} {test_at[other]:.4f} | '
              f'{time.time() - t_start:.0f}s', flush=True)

    scale = 100 if task == 'binary_auc' else 1      # report ROC-AUC in percent
    return {'seed': seed, 'num_aug': num_aug,
            f'test@{args.select}': scale * test_at[args.select], f'best_{args.select}': scale * best[args.select],
            f'test@{other}': scale * test_at[other], f'best_{other}': scale * best[other]}


def main():
    parser = argparse.ArgumentParser(description='CODA downstream OOD graph classification')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS))
    parser.add_argument('--data_dir', default='../data/GOOD', help='root of the GOOD datasets')
    parser.add_argument('--aug', default=None, help='generated set (.pkl of graphs or .txt of SMILES); omit for ERM')
    parser.add_argument('--select', default='val', choices=['val', 'id_val'],
                        help='validation split used for checkpoint selection (val = OOD validation)')
    parser.add_argument('--test_split', default='test', choices=['test', 'id_test'],
                        help='test = OOD test (default); id_test = in-distribution test')
    parser.add_argument('--seed', type=int, default=None, help='first seed; trial k uses seed + 10 * k')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default=None, help='write per-seed results to this JSON file')
    parser.add_argument('--save_ckpt', default=None,
                        help='save the best model (by the selection split) here; with several trials the seed is appended')
    # model / optimisation; defaults are the per-dataset settings of the paper (see data.DATASETS)
    parser.add_argument('--backbone', default=None, choices=['GIN', 'vGIN', 'GCN', 'GAT'])
    parser.add_argument('--model_layer', type=int, default=None, help='number of GNN layers')
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--test_epoch', type=int, default=None, help='checkpoints are selected after this epoch')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--milestones', nargs='*', type=int, default=None,
                        help='epochs at which the learning rate is multiplied by --lr_gamma (no values: no decay)')
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--var_weight', type=float, default=None,
                        help='weight of the variance of the mean losses on original and on generated graphs, '
                             'added to the loss (V-REx penalty; 0 = off)')
    # generate-then-filter baselines
    parser.add_argument('--ood_filter', default=None, choices=['density', 'mahalanobis'],
                        help='GDSS-CCD (density) / GDSS-CCM (mahalanobis) post-hoc filter of the generated set')
    parser.add_argument('--ood_filter_keep_ratio', type=float, default=0.5)
    parser.add_argument('--ood_filter_per_class', action='store_true',
                        help='keep the top fraction within every class instead of over the whole generated set')
    parser.add_argument('--pretrain_ckpt', default=None,
                        help='encoder checkpoint (saved with --save_ckpt) for the OOD filter embeddings; '
                             'default: a randomly initialised encoder')
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    for key in ('backbone', 'model_layer', 'epochs', 'milestones', 'test_epoch', 'batch_size', 'lr', 'var_weight', 'seed'):
        if getattr(args, key) is None:
            setattr(args, key, spec[key])
    if args.epochs <= args.test_epoch:
        parser.error(f'--epochs ({args.epochs}) must exceed --test_epoch ({args.test_epoch}): '
                     'checkpoints are only selected after test_epoch')
    if args.ood_filter and not args.aug:
        parser.error('--ood_filter needs a generated set (--aug)')
    print(json.dumps(vars(args), indent=2), flush=True)

    results = []
    for k in range(args.trials):
        results.append(run(args, args.seed + 10 * k))
        print('RESULT', json.dumps(results[-1]), flush=True)
        if args.out:                                # written after every trial
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump({'args': vars(args), 'results': results}, f, indent=2)
    key = f'test@{args.select}'
    scores = [r[key] for r in results]
    print(f'Final {key}: {np.mean(scores):.2f} ± {np.std(scores):.2f} over {len(scores)} seed(s)')


if __name__ == '__main__':
    main()
