from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .colmap import image_index, qvec2rotmat, read_colmap
from .config import load_scene_config, resolve_path
from .geometry import BoundingBox, Camera, Trajectory


@dataclass
class SceneMetadata:
    config: object
    root: Path
    timestamp_range: list[int]
    cameras: dict[str, Camera]
    ego_poses: Trajectory
    bounding_boxes: dict[str, BoundingBox]
    static_bounding_boxes: dict[str, BoundingBox]


def _camera_matrix(camera):
    if camera.model == "SIMPLE_PINHOLE":
        return torch.tensor(((camera.params[0], 0, camera.params[1]), (0, camera.params[0], camera.params[2]), (0, 0, 1)), dtype=torch.float32)
    if camera.model in {"PINHOLE", "OPENCV"}:
        return torch.tensor(((camera.params[0], 0, camera.params[2]), (0, camera.params[1], camera.params[3]), (0, 0, 1)), dtype=torch.float32)
    raise ValueError(f"Unsupported camera model: {camera.model}")


def _load_cameras(colmap_path, timestamp_range, involved_sensors):
    cameras, images = read_colmap(colmap_path)
    indexed = image_index(images)
    start, end = timestamp_range
    result = {}
    for name in involved_sensors:
        timestamps = [timestamp for timestamp in sorted(indexed[name]) if start <= timestamp < end]
        first_image = indexed[name][timestamps[0]]
        colmap_camera = cameras[first_image.camera_id]
        camera = Camera(name, colmap_camera.height, colmap_camera.width, _camera_matrix(colmap_camera))
        for timestamp in timestamps:
            image = indexed[name][timestamp]
            w2c = torch.eye(4)
            w2c[:3, :3] = torch.tensor(qvec2rotmat(image.qvec), dtype=torch.float32)
            w2c[:3, 3] = torch.tensor(image.tvec, dtype=torch.float32)
            camera.add_ext(timestamp, w2c)
        result[name] = camera
    return result


def _load_ego_poses(data_dir, timestamp_range, test_interval):
    with (data_dir / "ego_pose.json").open() as file:
        raw_poses = json.load(file)
    start, end = timestamp_range
    poses = {}
    counter = 0
    for key, value in sorted(raw_poses.items(), key=lambda item: int(item[0])):
        timestamp = int(key)
        if timestamp < start or timestamp >= end:
            continue
        counter += 1
        if counter % test_interval == 0:
            continue
        poses[timestamp] = torch.tensor(value, dtype=torch.float32).reshape(4, 4)
    return Trajectory(poses)


def _load_bounding_boxes(data_dir, timestamp_range, test_interval, dynamic_speed):
    with (data_dir / "bounding_box.json").open() as file:
        raw_boxes = json.load(file)
    start, end = timestamp_range
    dynamic = {}
    static = {}
    for identifier, raw_box in raw_boxes.items():
        box = BoundingBox(identifier, raw_box["type"], raw_box["size"])
        for index, (key, value) in enumerate(raw_box["local2world"].items()):
            timestamp = int(key)
            if not start <= timestamp <= end or (index + 1) % test_interval == 0:
                continue
            box.add_timestamp(timestamp, torch.tensor(value, dtype=torch.float32).reshape(4, 4))
        if len(box) <= 1:
            continue
        velocities = []
        for first, second in zip(box.get_sorted_ts()[:-1], box.get_sorted_ts()[1:]):
            velocities.append(torch.norm(box.get_translation(second) - box.get_translation(first)).item() / ((second - first) * 10e-6))
        if sum(velocities) / len(velocities) > dynamic_speed or box.object_type == "nonrigid":
            dynamic[identifier] = box
        else:
            static[identifier] = box
    return dynamic, static


def load_scene_metadata(scene_id: str):
    config, root = load_scene_config(scene_id)
    dataset = config.dataloader_cfg.dataset_cfg
    timestamp_range = [int(timestamp) for timestamp in dataset.timestamp_range]
    data_dir = resolve_path(root, dataset.data_dir)
    cameras = _load_cameras(resolve_path(root, dataset.colmap_path), timestamp_range, dataset.involved_sensors)
    dynamic, static = _load_bounding_boxes(data_dir, timestamp_range, int(dataset.test_interval), float(dataset.dynamic_speed))
    ego_poses = _load_ego_poses(data_dir, timestamp_range, int(dataset.test_interval))
    return SceneMetadata(config, root, timestamp_range, cameras, ego_poses, dynamic, static)
