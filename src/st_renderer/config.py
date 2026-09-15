from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _wrap(value):
    if isinstance(value, dict):
        return AttrDict({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _config_root(scene_path: Path) -> Path:
    for parent in scene_path.parents:
        if parent.name == "configs":
            return parent.parent
    raise ValueError(f"Scene config must be under a configs directory: {scene_path}")


def _load(path: Path, root: Path):
    with path.open() as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {path}")

    merged = {}
    for base in config.pop("configs", []):
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = root / base_path
        merged = _merge(merged, _load(base_path, root))
    return _merge(merged, config)


def load_scene_config(scene_id: str):
    path = Path(scene_id).resolve()
    root = _config_root(path)
    return _wrap(_load(path, root)), root


def resolve_path(root: Path, value: str | None):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path
