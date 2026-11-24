"""CODA generation stage.

    python main.py sample           --config configs/motif_basis.yaml --out samples/motif_basis/lam0.04
    python main.py train_score      --config configs/motif_basis.yaml --out_dir runs/motif_basis
    python main.py train_classifier --config configs/motif_basis.yaml --out_dir runs/motif_basis

Any config entry can be overridden with `--set section.key=value`, e.g. `--set sampler.ood=0.02`.
"""
import argparse

from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='CODA: score-based conditional OOD graph augmentation')
    stages = parser.add_subparsers(dest='stage', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', required=True, help='path to a dataset config (configs/*.yaml)')
    common.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')

    sample = stages.add_parser('sample', parents=[common], help='generate an augmentation set')
    sample.add_argument('--out', required=True, help='output prefix of the generated set, e.g. samples/motif_basis/lam0.04')
    for name, what in (('train_score', 'score network'), ('train_classifier', 'time-dependent classifier')):
        train = stages.add_parser(name, parents=[common], help=f'train the {what}')
        train.add_argument('--out_dir', required=True, help='directory that receives the checkpoints')
        train.add_argument('--seed', type=int, default=42, help='training seed')
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    if args.stage == 'sample':
        from sampler import Sampler
        Sampler(config, args.out).sample()
    else:
        from trainer import ClassifierTrainer, ScoreTrainer
        config.seed = args.seed
        trainer = ScoreTrainer if args.stage == 'train_score' else ClassifierTrainer
        trainer(config, args.out_dir).train()


if __name__ == '__main__':
    main()
