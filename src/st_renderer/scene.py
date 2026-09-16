from __future__ import annotations

import json
import math

import torch
from fast_gauss import BatchedGaussianRasterizer, GaussianRasterizationSettings

from .gaussians import GaussianBackground, GaussianModelNonrigid, GaussianModelRigid, SkyCubeMap


def _raster_settings(camera, background):
    return GaussianRasterizationSettings(
        image_height=camera["image_height"],
        image_width=camera["image_width"],
        tanfovx=math.tan(camera["FoVx"] * 0.5),
        tanfovy=math.tan(camera["FoVy"] * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera["world_view_transform"],
        projmatrix=camera["full_proj_transform"],
        campos=camera["camera_center"],
        prefiltered=False,
    )


class StreetGaussianScene:
    def __init__(self, config, timestamp_range, cameras, bounding_boxes):
        self.timestamp_range = timestamp_range
        self.camera_sensors = cameras
        self.bkgd_max_sh_degree = config.bkgd_gaussian_cfg.max_sh_degree
        self.rigid_max_sh_degree = config.rigid_gaussian_cfg.max_sh_degree
        self.rigid_max_sh_degree_t = config.rigid_gaussian_cfg.max_sh_degree_t
        self.nonrigid_max_sh_degree = config.nonrigid_gaussian_cfg.max_sh_degree
        self.nonrigid_max_sh_degree_t = config.nonrigid_gaussian_cfg.max_sh_degree_t
        self.bkgd_active_sh_degree = 0
        self.rigid_active_sh_degree = 0
        self.rigid_active_sh_degree_t = 0
        self.nonrigid_active_sh_degree = 0
        self.nonrigid_active_sh_degree_t = 0
        self.background = torch.tensor(config.background, device="cuda", dtype=torch.float32)
        self.sh_sigmoid_warmup = config.sh_sigmoid_warmup
        self.is_dynamic = config.is_dynamic

        self.bkgd_gaussians = GaussianBackground(self.bkgd_max_sh_degree)
        self.rigid_gaussians = []
        self.nonrigid_gaussians = []
        if self.is_dynamic:
            for bounding_box in bounding_boxes.values():
                if bounding_box.object_type == "rigid":
                    self.rigid_gaussians.append(GaussianModelRigid(bounding_box, self.rigid_max_sh_degree, self.rigid_max_sh_degree_t))
                elif bounding_box.object_type == "nonrigid":
                    self.nonrigid_gaussians.append(GaussianModelNonrigid(bounding_box, self.nonrigid_max_sh_degree, self.nonrigid_max_sh_degree_t))
        if not self.object_gaussians:
            self.is_dynamic = False
        if config.skymap_cfg.use_sky_cubemap:
            self.sky_cubemap = SkyCubeMap(config.skymap_cfg.resolution)

    @property
    def object_gaussians(self):
        return self.rigid_gaussians + self.nonrigid_gaussians

    def load_model(self, model_paths):
        self.bkgd_active_sh_degree = self.bkgd_max_sh_degree
        self.rigid_active_sh_degree = self.rigid_max_sh_degree
        self.rigid_active_sh_degree_t = self.rigid_max_sh_degree_t
        self.nonrigid_active_sh_degree = self.nonrigid_max_sh_degree
        self.nonrigid_active_sh_degree_t = self.nonrigid_max_sh_degree_t
        if model_paths.get("background_ply_path"):
            self.bkgd_gaussians.load_ply(model_paths["background_ply_path"])
        if model_paths.get("objects_pth_path"):
            states = torch.load(model_paths["objects_pth_path"])
            for gaussian, state in zip(self.object_gaussians, states):
                gaussian.load(state)
        if model_paths.get("sky_cubemap_path"):
            self.sky_cubemap.load(torch.load(model_paths["sky_cubemap_path"]))
        if model_paths.get("object_offset_path"):
            with open(model_paths["object_offset_path"]) as file:
                offsets = json.load(file)
            for gaussian in self.object_gaussians:
                if gaussian.id in offsets:
                    gaussian.bounding_box.import_offset(offsets[gaussian.id])
        if model_paths.get("camera_offset_path"):
            with open(model_paths["camera_offset_path"]) as file:
                offsets = json.load(file)
            for name, camera in self.camera_sensors.items():
                camera.import_offset(offsets[name])
        self._pack_gaussians()

    def _pack_gaussians(self):
        gaussians = [self.bkgd_gaussians, *self.object_gaussians]
        xyz = []
        scales = []
        rotations = []
        opacities = []
        ts = []
        scale_ts = []
        rotation_rs = []
        gid_map = []
        object_sources = torch.empty((len(self.object_gaussians), 2), device="cuda", dtype=torch.int32)
        gid_start = 0
        rigid_feature_start = 0
        nonrigid_feature_start = 0
        for object_id, gaussian in enumerate(gaussians):
            amount = gaussian.amount
            xyz.append(gaussian.get_xyz())
            scales.append(gaussian.get_scaling())
            rotations.append(gaussian.get_quaternion())
            opacities.append(gaussian.get_opacity().reshape(-1))
            mapping = torch.empty((amount, 2), device="cuda", dtype=torch.int32)
            if gaussian is self.bkgd_gaussians:
                mapping[:, 0] = 0
                mapping[:, 1] = -1
                ts.append(torch.zeros((amount,), device="cuda"))
                scale_ts.append(torch.ones((amount,), device="cuda"))
                rotation_rs.append(torch.zeros((amount, 4), device="cuda"))
            elif isinstance(gaussian, GaussianModelRigid):
                mapping[:, 0] = 1
                mapping[:, 1] = object_id - 1
                object_sources[object_id - 1, 0] = gid_start
                object_sources[object_id - 1, 1] = rigid_feature_start
                rigid_feature_start += amount
                ts.append(torch.zeros((amount,), device="cuda"))
                scale_ts.append(torch.ones((amount,), device="cuda"))
                rotation_rs.append(torch.zeros((amount, 4), device="cuda"))
            else:
                mapping[:, 0] = 2
                mapping[:, 1] = object_id - 1
                object_sources[object_id - 1, 0] = gid_start
                object_sources[object_id - 1, 1] = nonrigid_feature_start
                nonrigid_feature_start += amount
                ts.append(gaussian.get_t().reshape(-1))
                scale_ts.append(gaussian.get_scaling_t().reshape(-1))
                rotation_rs.append(gaussian.get_quaternion_r())
            gid_map.append(mapping)
            gid_start += amount
        self._raw_gaussians = {
            "xyz": torch.cat(xyz).contiguous(),
            "scales": torch.cat(scales).contiguous(),
            "rotations": torch.cat(rotations).contiguous(),
            "opacities": torch.cat(opacities).contiguous(),
            "background_features": self.bkgd_gaussians.get_features(),
            "rigid_features": torch.cat([gaussian.get_features() for gaussian in self.rigid_gaussians]).contiguous() if self.rigid_gaussians else torch.empty((0, (self.rigid_max_sh_degree + 1) ** 2 + self.rigid_max_sh_degree_t, 3), device="cuda"),
            "nonrigid_features": torch.cat([gaussian.get_features() for gaussian in self.nonrigid_gaussians]).contiguous() if self.nonrigid_gaussians else torch.empty((0, (self.nonrigid_max_sh_degree + 1) ** 2 * (self.nonrigid_max_sh_degree_t + 1), 3), device="cuda"),
            "ts": torch.cat(ts).contiguous(),
            "scale_ts": torch.cat(scale_ts).contiguous(),
            "rotation_rs": torch.cat(rotation_rs).contiguous(),
            "gid_map": torch.cat(gid_map).contiguous(),
            "object_sources": object_sources,
            "background_feature_stride": (self.bkgd_max_sh_degree + 1) ** 2,
            "rigid_feature_stride": (self.rigid_max_sh_degree + 1) ** 2 + self.rigid_max_sh_degree_t,
            "nonrigid_feature_stride": (self.nonrigid_max_sh_degree + 1) ** 2 * (self.nonrigid_max_sh_degree_t + 1),
            "background_degree": self.bkgd_active_sh_degree,
            "rigid_degree": self.rigid_active_sh_degree,
            "rigid_degree_t": self.rigid_active_sh_degree_t,
            "rigid_max_degree_t": self.rigid_max_sh_degree_t,
            "nonrigid_degree": self.nonrigid_active_sh_degree,
            "nonrigid_degree_t": self.nonrigid_active_sh_degree_t,
        }

    def get_gaussian_attributes(self, timestamp, extra_bboxes):
        object_count = len(self.object_gaussians)
        object_poses = torch.empty((object_count, 4, 4), device="cuda")
        object_active = torch.zeros((object_count,), device="cuda", dtype=torch.int32)
        for object_id, gaussian in enumerate(self.object_gaussians):
            if gaussian.id in extra_bboxes:
                object_poses[object_id].copy_(extra_bboxes[gaussian.id])
                object_active[object_id] = 1
        return self._raw_gaussians | {
            "object_poses": object_poses,
            "object_active": object_active,
            "timestamp": (timestamp - self.timestamp_range[0]) / (self.timestamp_range[1] - self.timestamp_range[0]),
        }

    @torch.no_grad()
    def render_batch(self, timestamp, camera_params, ray_d, extra_bboxes):
        raw = self.get_gaussian_attributes(timestamp, extra_bboxes)
        rasterizer = BatchedGaussianRasterizer(
            [_raster_settings(camera, self.background) for camera in camera_params],
            init_buffer_size=raw["xyz"].shape[0],
            init_texture_size=[camera_params[0]["image_height"], camera_params[0]["image_width"]],
        )
        rgb, alpha = rasterizer.render_street(raw)
        if hasattr(self, "sky_cubemap"):
            rgb = torch.lerp(self.sky_cubemap(ray_d), rgb, alpha)
        return rgb.detach().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
