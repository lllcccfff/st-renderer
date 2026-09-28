from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .geometry import BoundingBox, Trajectory


FORMAT_VERSION = 1
_DYNAMIC_SPEED = {"nuscenes": 0.01, "waymo": 0.05}


@dataclass
class SceneMetadata:
    dataset: str
    scene_name: str
    map_id: str | None
    timestamp_range: list[int]
    cameras: dict[str, dict]
    ego_poses: Trajectory
    bounding_boxes: dict[str, BoundingBox]
    static_bounding_boxes: dict[str, BoundingBox]


def load_json_member(archive, key):
    return json.loads(archive[key].tobytes().decode("utf-8"))


def validate_archive_version(archive):
    version = archive["format_version"].item()
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported archive format_version: {version}")


def _load_ego_poses(raw_poses, timestamp_range):
    start, end = timestamp_range
    poses = {}
    for key, value in sorted(raw_poses.items(), key=lambda item: int(item[0])):
        timestamp = int(key)
        if timestamp < start or timestamp >= end:
            continue
        poses[timestamp] = torch.tensor(value, dtype=torch.float32).reshape(4, 4)
    return Trajectory(poses)


def _load_bounding_boxes(raw_boxes, timestamp_range, dynamic_speed):
    start, end = timestamp_range
    dynamic = {}
    static = {}
    for identifier, raw_box in raw_boxes.items():
        box = BoundingBox(identifier, raw_box["type"], raw_box["size"])
        for key, value in raw_box["local2world"].items():
            timestamp = int(key)
            if not start <= timestamp <= end:
                continue
            box.add_timestamp(timestamp, torch.tensor(value, dtype=torch.float32).reshape(4, 4))
        if len(box) <= 1:
            continue
        velocities = []
        for first, second in zip(box.get_sorted_ts()[:-1], box.get_sorted_ts()[1:]):
            distance = torch.norm(box.get_translation(second) - box.get_translation(first)).item()
            velocities.append(distance / ((second - first) * 10e-6))
        if sum(velocities) / len(velocities) > dynamic_speed or box.object_type == "nonrigid":
            dynamic[identifier] = box
        else:
            static[identifier] = box
    return dynamic, static


def _apply_object_offsets(boxes, offsets):
    for identifier, box in boxes.items():
        if identifier in offsets:
            box.import_offset(offsets[identifier])


def _load_camera_parameters(raw_cameras):
    return {
        name: {
            "K": np.asarray(camera["K"], dtype=np.float32).tolist(),
            "H": int(camera["H"]),
            "W": int(camera["W"]),
            "ego2camera": np.asarray(camera["ego2camera"], dtype=np.float32).tolist(),
        }
        for name, camera in raw_cameras.items()
    }


def load_scene_metadata(path: str | Path):
    with np.load(path, allow_pickle=False) as archive:
        validate_archive_version(archive)
        scene = load_json_member(archive, "scene")
        cameras = _load_camera_parameters(load_json_member(archive, "camera"))
        raw_ego_poses = load_json_member(archive, "ego_pose")
        raw_boxes = load_json_member(archive, "bounding_box")
        offsets = load_json_member(archive, "object_offset")

    dataset = scene["dataset"]
    scene_name = str(scene["scene_name"])
    timestamp_range = [int(timestamp) for timestamp in scene["timestamp_range"]]

    dynamic, static = _load_bounding_boxes(raw_boxes, timestamp_range, _DYNAMIC_SPEED[dataset])
    _apply_object_offsets(dynamic | static, offsets)

    return SceneMetadata(
        dataset=dataset,
        scene_name=scene_name,
        map_id=scene["map_id"],
        timestamp_range=timestamp_range,
        cameras=cameras,
        ego_poses=_load_ego_poses(raw_ego_poses, timestamp_range),
        bounding_boxes=dynamic,
        static_bounding_boxes=static,
    )
