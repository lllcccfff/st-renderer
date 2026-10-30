from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from .geometry import get_fov
from .metadata import load_json_member, load_scene_metadata, validate_archive_version
from .scene import StreetGaussianScene


_DEFAULT_ROOTS = {
    "nuscenes": Path("data/benchmark/nuscenes"),
    "waymo": Path("data/benchmark/waymo"),
}


def _cuda_async(tensor):
    return tensor if tensor.is_cuda else tensor.pin_memory().cuda(non_blocking=True)


class SimulatorInterface:
    def __init__(
        self,
        dataset,
        root=None,
        zNear=0.0001,
        zFar=1000,
    ):
        self.dataset = dataset
        self.root = Path(_DEFAULT_ROOTS[dataset] if root is None else root).expanduser().resolve()
        self.zNear = zNear
        self.zFar = zFar
        self.scene_data = {}
        self._maps = {}
        self.current_timestamp = None
        self.current_objects = {}
        self._camera_parameters = None

    def _scene_path(self, scene_name):
        return self.root / f"{scene_name}.npz"

    def load_metadata(self, scene_name):
        scene_name = str(scene_name)
        metadata = load_scene_metadata(self._scene_path(scene_name))
        self.scene_data[scene_name] = metadata
        trackings = metadata.bounding_boxes | metadata.static_bounding_boxes
        ground = self.root / "ground" / f"{scene_name}.obj" if self.dataset == "waymo" else None
        return (
            list(metadata.timestamp_range),
            metadata.cameras,
            self._parse_ego_pose_data(metadata.ego_poses, *metadata.timestamp_range),
            self._parse_tracking_data(trackings),
            str(ground) if ground is not None else None,
            None,
        )

    def load_model(self, scene_name):
        scene_name = str(scene_name)
        self._camera_parameters = None
        metadata = self.scene_data[scene_name]
        with np.load(self._scene_path(scene_name), allow_pickle=False) as archive:
            validate_archive_version(archive)
            background = io.BytesIO(archive["background"].tobytes())
            objects = io.BytesIO(archive["objects"].tobytes())
            skymap = io.BytesIO(archive["skymap"].tobytes())

        self.scene_model = StreetGaussianScene(metadata.timestamp_range, metadata.bounding_boxes)
        self.scene_model.load_model(background, objects, skymap)
        if metadata.map_id is None:
            return None
        return self._load_map(metadata.map_id)

    def _load_map(self, map_id):
        if map_id in self._maps:
            return self._maps[map_id]

        from trajdata.maps.vec_map import VectorMap
        from trajdata.proto.vectorized_map_pb2 import VectorizedMap

        with np.load(self.root / "map_cache.npz", allow_pickle=False) as archive:
            validate_archive_version(archive)
            manifest = load_json_member(archive, "manifest")
            map_key = manifest[map_id]
            payload = archive[map_key].tobytes()

        proto = VectorizedMap()
        proto.ParseFromString(payload)
        vector_map = VectorMap.from_proto(
            proto,
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_roads=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True,
        )
        vector_map.compute_search_indices()
        self._maps[map_id] = vector_map
        return vector_map

    def update_scene(self, timestamp, object_poses):
        self.current_timestamp = timestamp
        self.current_objects = {identifier: _cuda_async(pose) for identifier, pose in object_poses.items()}

    def close(self):
        import fast_gauss

        fast_gauss.close()
        self.scene_model = None
        self.current_objects = {}
        self._camera_parameters = None

    def _init_camera_parameters(self, K, H, W):
        fovs = [get_fov(camera_K, height, width) for camera_K, height, width in zip(K, H, W)]
        K = _cuda_async(K)
        widths = torch.as_tensor(W, dtype=K.dtype, device=K.device)
        heights = torch.as_tensor(H, dtype=K.dtype, device=K.device)
        projections = torch.zeros((len(H), 4, 4), dtype=K.dtype, device=K.device)
        projections[:, 0, 0] = 2 * K[:, 0, 0] / widths
        projections[:, 0, 1] = 2 * K[:, 0, 1] / widths
        projections[:, 0, 2] = -1 + 2 * K[:, 0, 2] / widths
        projections[:, 1, 1] = 2 * K[:, 1, 1] / heights
        projections[:, 1, 2] = -1 + 2 * K[:, 1, 2] / heights
        projections[:, 2, 2] = (self.zFar + self.zNear) / (self.zFar - self.zNear)
        projections[:, 2, 3] = -2 * self.zFar * self.zNear / (self.zFar - self.zNear)
        projections[:, 3, 2] = 1
        height, width = H[0], W[0]
        x, y = torch.meshgrid(
            torch.arange(width, dtype=K.dtype, device=K.device),
            torch.arange(height, dtype=K.dtype, device=K.device),
            indexing="xy",
        )
        pixels = torch.stack((x + .5, y + .5, torch.ones_like(x)), dim=-1).reshape(1, -1, 3)
        rays = pixels @ torch.linalg.inv(K).transpose(-1, -2)
        self._camera_parameters = projections, rays / torch.norm(rays, dim=-1, keepdim=True), fovs

    def render(self, K, H, W, extrinsics):
        # Intrinsics are fixed for the loaded camera rig, like its pixel rays.
        if self._camera_parameters is None:
            self._init_camera_parameters(K, H, W)
        projections, camera_rays, fovs = self._camera_parameters
        w2c = _cuda_async(extrinsics).contiguous()
        height, width = H[0], W[0]
        rays = (camera_rays @ w2c[:, :3, :3]).reshape(len(H), height, width, 3)
        world_view = w2c.transpose(-1, -2)
        full_projection = (projections @ w2c).transpose(-1, -2).contiguous()
        centers = torch.linalg.inv(w2c)[:, :3, 3].contiguous()
        cameras = [
            {
                "world_view_transform": world_view[camera_id],
                "full_proj_transform": full_projection[camera_id],
                "camera_center": centers[camera_id],
                "image_height": height,
                "image_width": width,
                "FoVx": fovs[camera_id][0],
                "FoVy": fovs[camera_id][1],
            }
            for camera_id in range(len(H))
        ]
        return self.scene_model.render_batch(self.current_timestamp, cameras, rays, self.current_objects)

    @staticmethod
    def _parse_ego_pose_data(trajectory, start_timestamp, end_timestamp):
        return {
            timestamp: trajectory.get_transform(timestamp).tolist()
            for timestamp in trajectory._sorted_ts
            if start_timestamp <= timestamp <= end_timestamp
        }

    @staticmethod
    def _parse_tracking_data(trackings):
        return {
            identifier: {
                "poses": {
                    timestamp: tracking.get_transform(timestamp).tolist()
                    for timestamp in tracking.get_sorted_ts()
                },
                "size": tracking.size.tolist(),
                "type": (
                    "vehicle"
                    if tracking.detail_type == "rigid"
                    else "pedestrian"
                    if tracking.detail_type == "nonrigid"
                    else tracking.detail_type
                ),
            }
            for identifier, tracking in trackings.items()
        }
