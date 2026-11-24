"""YAML configuration with dotted command-line overrides."""
import ast

import yaml
from easydict import EasyDict as edict


def _parse(value):
    """`0.04` -> float, `[1, 1, 1]` -> list, `true` / `None` -> bool / None, anything else stays a string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {'true': True, 'false': False, 'none': None, 'null': None}.get(value.lower(), value)


def load_config(path, overrides=()):
    """Read `path` and apply overrides such as `sampler.ood=0.03` or `sample.n_graphs=1000`.

    An override must name an existing entry, so that a misspelt key cannot be ignored silently.
    """
    with open(path, 'r') as f:
        config = edict(yaml.safe_load(f))
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'override "{item}" is not of the form section.key=value')
        node = config
        *parents, leaf = key.split('.')
        for name in parents + [leaf]:
            if not isinstance(node, dict) or name not in node:
                raise KeyError(f'unknown config entry "{key}" (in {path})')
            parent, node = node, node[name]
        parent[leaf] = _parse(value)
    return config


def to_plain(obj):
    """Recursively convert EasyDict / tuples into plain python containers (safe to store in checkpoints)."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj
