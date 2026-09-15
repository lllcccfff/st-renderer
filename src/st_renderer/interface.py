from __future__ import annotations

import json

import torch

from .config import resolve_path
from .geometry import gen_rays, get_fov, get_projection_matrix
from .metadata import load_scene_metadata
from .scene import StreetGaussianScene


def _cuda_async(tensor):
    return tensor if tensor.is_cuda else tensor.pin_memory().cuda(non_blocking=True)


class SimulatorInterface:
    def __init__(self, zNear=0.0001, zFar=1000):
        self.zNear = zNear
        self.zFar = zFar
        self.scene_data = {}
        self.current_timestamp = None
        self.current_objects = {}

    def load_metadata(self, scene_id):
        scene_id = str(scene_id)
        metadata = load_scene_metadata(scene_id)
        self.scene_data[scene_id] = metadata
        config = metadata.config
        model_paths = config.visualizer_cfg.model_path
        trackings = metadata.bounding_boxes | metadata.static_bounding_boxes
        tracking_data = self._parse_tracking_data(trackings, metadata.timestamp_range[0], resolve_path(metadata.root, model_paths.get("object_offset_path")))
        background_mesh = resolve_path(metadata.root, config.dataloader_cfg.dataset_cfg.data_dir) / "background.obj"
        if not background_mesh.exists():
            background_mesh = None
        if "hd_map" in config.dataloader_cfg and str(config.dataloader_cfg.hd_map.map_type).lower() == "nuscenes":
            self._load_map(scene_id, metadata)
        return (
            list(metadata.timestamp_range),
            self._parse_camera_data(metadata.cameras, metadata.ego_poses, metadata.timestamp_range[0]),
            self._parse_ego_pose_data(metadata.ego_poses, *metadata.timestamp_range),
            tracking_data,
            str(background_mesh) if background_mesh else None,
            None,
        )

    def _load_map(self, scene_id, metadata):
        from trajdata.maps import MapAPI

        hd_map = metadata.config.dataloader_cfg.hd_map
        version = getattr(hd_map, "nuscene_version", "v1.0-trainval")
        nuscene_root = resolve_path(metadata.root, hd_map.nuscene_root)
        table_dir = nuscene_root / version
        scenes = json.loads((table_dir / "scene.json").read_text())
        logs = json.loads((table_dir / "log.json").read_text())
        scene_name = str(metadata.config.dataloader_cfg.dataset_cfg.scene_name)
        nuscene_id = "scene-" + scene_name.rsplit("_", 1)[0]
        scene = {item["name"]: item for item in scenes}[nuscene_id]
        location = {item["token"]: item for item in logs}[scene["log_token"]]["location"]
        self.scene_data[scene_id].map_name = f"nusc_{version.split('-')[1]}:{location}"
        if not hasattr(self, "map_api"):
            self.map_api = MapAPI(metadata.root / "data/.unified_data_cache")

    def load_model(self, scene_id):
        metadata = self.scene_data[str(scene_id)]
        self.scene_model = StreetGaussianScene(
            metadata.config.model_cfg,
            metadata.timestamp_range,
            metadata.cameras,
            metadata.bounding_boxes,
        )
        self.scene_model.load_model(
            {key: resolve_path(metadata.root, value) for key, value in metadata.config.visualizer_cfg.model_path.items()}
        )
        if hasattr(self, "map_api"):
            return self.map_api.get_map(
                metadata.map_name,
                incl_centerline=True,
                incl_lane_edges=True,
                incl_lane_area=True,
                incl_ped_crosswalks=True,
                incl_ped_walkways=True,
            )

    def update_scene(self, timestamp, object_poses):
        self.current_timestamp = timestamp
        self.current_objects = {identifier: _cuda_async(pose) for identifier, pose in object_poses.items()}

    def render(self, K, H, W, extrinsics):
        fovs = [get_fov(camera_K, height, width) for camera_K, height, width in zip(K, H, W)]
        projections = [get_projection_matrix(camera_K, height, width, self.zNear, self.zFar) for camera_K, height, width in zip(K, H, W)]
        K = _cuda_async(K)
        w2c = _cuda_async(extrinsics).contiguous()
        cameras = []
        rays = []
        for camera_K, camera_w2c, height, width, fov, projection in zip(K, w2c, H, W, fovs, projections):
            projection = _cuda_async(projection)
            cameras.append(
                {
                    "world_view_transform": camera_w2c.T,
                    "full_proj_transform": (projection @ camera_w2c).contiguous().T,
                    "camera_center": camera_w2c.inverse()[:3, 3].contiguous(),
                    "image_height": height,
                    "image_width": width,
                    "FoVx": fov[0],
                    "FoVy": fov[1],
                }
            )
            rays.append(gen_rays(camera_w2c[:3, :3].T, camera_w2c[:3, 3], camera_K, height, width)[1])
        return self.scene_model.render_batch(self.current_timestamp, cameras, torch.stack(rays), self.current_objects)

    @staticmethod
    def _parse_ego_pose_data(trajectory, start_timestamp, end_timestamp):
        return {
            timestamp: trajectory.get_transform(timestamp).tolist()
            for timestamp in trajectory._sorted_ts
            if start_timestamp <= timestamp <= end_timestamp
        }

    @staticmethod
    def _parse_camera_data(cameras, ego_poses, _):
        parsed = {}
        for name, camera in cameras.items():
            timestamp = next(iter(camera.world_view_transform))
            ego2world = ego_poses.get_transform(timestamp).cuda(non_blocking=True)
            parsed[name] = {
                "K": camera.K.tolist(),
                "H": camera.image_height,
                "W": camera.image_width,
                "ego2camera": (camera.world_view_transform[timestamp].T @ ego2world).tolist(),
            }
        return parsed

    @staticmethod
    def _parse_tracking_data(trackings, _, object_offset_path):
        if object_offset_path and object_offset_path.exists():
            with object_offset_path.open() as file:
                offsets = json.load(file)
            for identifier, tracking in trackings.items():
                if identifier in offsets:
                    tracking.import_offset(offsets[identifier])
        parsed = {}
        for identifier, tracking in trackings.items():
            parsed[identifier] = {
                "poses": {timestamp: tracking.get_transform(timestamp).tolist() for timestamp in tracking.get_sorted_ts()},
                "size": tracking.size.tolist(),
                "type": "vehicle" if tracking.detail_type == "rigid" else "pedestrian" if tracking.detail_type == "nonrigid" else tracking.detail_type,
            }
        return parsed
