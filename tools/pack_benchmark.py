#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from st_renderer.colmap import image_index, qvec2rotmat, read_colmap
from st_renderer.geometry import Trajectory


FORMAT_VERSION = 1
MAP_MANIFEST = {
    "nusc_trainval:boston-seaport": "boston_seaport",
    "nusc_trainval:singapore-onenorth": "singapore_onenorth",
    "nusc_trainval:singapore-queenstown": "singapore_queenstown",
    "nusc_trainval:singapore-hollandvillage": "singapore_hollandvillage",
}


def _merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path, root):
    config = yaml.safe_load(path.read_text()) or {}
    merged = {}
    for base in config.pop("configs", []):
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = root / base_path
        merged = _merge(merged, _load_config(base_path, root))
    return _merge(merged, config)


def load_scene_config(path):
    path = Path(path).expanduser().resolve()
    root = next(parent.parent for parent in path.parents if parent.name == "configs")
    return _load_config(path, root), root


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_model_path(config_root, model_root, value):
    path = Path(value)
    if model_root is None:
        return resolve_path(config_root, path)
    if path.is_absolute() or not path.parts or path.parts[0] != "output_shared":
        raise ValueError(f"Model path must be relative to output_shared: {path}")
    return Path(model_root).expanduser().resolve().joinpath(*path.parts[1:])


def encode_json(value):
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return np.frombuffer(payload, dtype=np.uint8)


def read_bytes(path):
    return np.fromfile(path, dtype=np.uint8)


def _camera_matrix(camera):
    if camera.model == "SIMPLE_PINHOLE":
        return np.array(
            ((camera.params[0], 0, camera.params[1]), (0, camera.params[0], camera.params[2]), (0, 0, 1)),
            dtype=np.float32,
        )
    if camera.model in {"PINHOLE", "OPENCV"}:
        return np.array(
            ((camera.params[0], 0, camera.params[2]), (0, camera.params[1], camera.params[3]), (0, 0, 1)),
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported camera model: {camera.model}")


def _load_camera_data(dataset, colmap_path, ego_pose_path, timestamp_range):
    cameras, images = read_colmap(colmap_path)
    indexed = image_index(images)
    raw_ego_poses = json.loads(ego_pose_path.read_bytes())
    start, end = timestamp_range
    ego_poses = Trajectory(
        {
            int(key): torch.tensor(value, dtype=torch.float32).reshape(4, 4)
            for key, value in raw_ego_poses.items()
            if start <= int(key) < end
        }
    )
    result = {}
    for name in dataset["involved_sensors"]:
        timestamps = [timestamp for timestamp in sorted(indexed[name]) if start <= timestamp < end]
        first_image = indexed[name][timestamps[0]]
        camera = cameras[first_image.camera_id]
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = torch.tensor(qvec2rotmat(first_image.qvec), dtype=torch.float32)
        w2c[:3, 3] = torch.tensor(first_image.tvec, dtype=torch.float32)
        ego2world = ego_poses.get_transform(timestamps[0])
        result[name] = {
            "K": _camera_matrix(camera).tolist(),
            "H": camera.height,
            "W": camera.width,
            "ego2camera": (w2c @ ego2world).tolist(),
        }
    return result


def _nuscenes_map_id(config, root, scene_name, nuscenes_raw_root=None):
    hd_map = config["dataloader_cfg"]["hd_map"]
    version = hd_map.get("nuscene_version", "v1.0-trainval")
    raw_root = (
        resolve_path(root, hd_map["nuscene_root"])
        if nuscenes_raw_root is None
        else Path(nuscenes_raw_root).expanduser().resolve()
    )
    table_dir = raw_root / version
    scenes = json.loads((table_dir / "scene.json").read_bytes())
    logs = json.loads((table_dir / "log.json").read_bytes())
    nuscene_name = "scene-" + scene_name.rsplit("_", 1)[0]
    scene = {item["name"]: item for item in scenes}[nuscene_name]
    location = {item["token"]: item for item in logs}[scene["log_token"]]["location"]
    return f"nusc_{version.split('-', 1)[1]}:{location}"


def pack_scene(
    scene_config,
    output_root,
    processed_root=None,
    model_root=None,
    nuscenes_raw_root=None,
):
    config, root = load_scene_config(scene_config)
    dataset_config = config["dataloader_cfg"]["dataset_cfg"]
    scene_name = str(dataset_config["scene_name"])
    dataset_name = config["exp_name"].removeprefix("benchmark_")
    timestamp_range = [int(timestamp) for timestamp in dataset_config["timestamp_range"]]
    data_dir = (
        resolve_path(root, dataset_config["data_dir"])
        if processed_root is None
        else Path(processed_root).expanduser().resolve() / scene_name
    )
    colmap_path = (
        resolve_path(root, dataset_config["colmap_path"])
        if processed_root is None
        else data_dir / "colmap"
    )
    configured_model_paths = config["visualizer_cfg"]["model_path"]
    model_paths = {
        key: resolve_model_path(root, model_root, value)
        for key, value in configured_model_paths.items()
    }
    model_dirs = {path.parent for path in model_paths.values()}
    if len(model_dirs) != 1:
        raise ValueError(f"Model files must share one directory: {sorted(map(str, model_dirs))}")
    model_dir = model_dirs.pop()
    if model_dir.parent.name != "models" or not model_dir.name.startswith("iteration_"):
        raise ValueError(f"Refusing to delete unexpected model directory: {model_dir}")
    ego_pose_path = data_dir / "ego_pose.json"
    scene = {
        "dataset": dataset_name,
        "scene_name": scene_name,
        "timestamp_range": timestamp_range,
        "map_id": (
            _nuscenes_map_id(config, root, scene_name, nuscenes_raw_root)
            if dataset_name == "nuscenes"
            else None
        ),
    }
    camera = _load_camera_data(dataset_config, colmap_path, ego_pose_path, timestamp_range)

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{scene_name}.npz"
    partial_path = output_path.with_suffix(".npz.partial")
    with partial_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            format_version=np.asarray(FORMAT_VERSION, dtype=np.int32),
            scene=encode_json(scene),
            camera=encode_json(camera),
            ego_pose=read_bytes(ego_pose_path),
            bounding_box=read_bytes(data_dir / "bounding_box.json"),
            object_offset=read_bytes(model_paths["object_offset_path"]),
            background=read_bytes(model_paths["background_ply_path"]),
            objects=read_bytes(model_paths["objects_pth_path"]),
            skymap=read_bytes(model_paths["sky_cubemap_path"]),
        )
    partial_path.replace(output_path)
    shutil.rmtree(model_dir)
    return dataset_name, root, output_path


def pack_map_cache(config_root, output_root):
    source_root = Path(config_root) / "data/.unified_data_cache/nusc_trainval/maps"
    members = {
        key: read_bytes(source_root / f"{map_id.split(':', 1)[1]}.pb")
        for map_id, key in MAP_MANIFEST.items()
    }
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "map_cache.npz"
    np.savez_compressed(
        output_path,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int32),
        manifest=encode_json(MAP_MANIFEST),
        **members,
    )
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(description="Pack StreetWorld benchmark scenes into NPZ archives")
    parser.add_argument("scene_configs", nargs="+", type=Path, help="Existing benchmark scene YAML files")
    parser.add_argument("--output-root", required=True, type=Path, help="Destination root for scene NPZ files")
    parser.add_argument("--processed-root", type=Path, help="Override root containing processed scene directories")
    parser.add_argument("--model-root", type=Path, help="Override the output_shared root")
    parser.add_argument("--nuscenes-raw-root", type=Path, help="Override the NuScenes raw dataset root")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    nuscenes_config_root = None
    for scene_config in args.scene_configs:
        dataset, config_root, output_path = pack_scene(
            scene_config,
            args.output_root,
            processed_root=args.processed_root,
            model_root=args.model_root,
            nuscenes_raw_root=args.nuscenes_raw_root,
        )
        print(output_path)
        if dataset == "nuscenes":
            nuscenes_config_root = config_root
    if nuscenes_config_root is not None:
        print(pack_map_cache(nuscenes_config_root, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
