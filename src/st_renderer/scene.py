from __future__ import annotations

import json
import math

import torch
from fast_gauss import BatchedGaussianRasterizer, GaussianRasterizationSettings
from mixplat.cov_utils import compute_cov3d, compute_cov3d_conditional
from mixplat.sh_utils import compute_sh, compute_sh_fourier

from .gaussians import GaussianBackground, GaussianModelNonrigid, GaussianModelRigid, SkyCubeMap
from .geometry import build_rotation


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


def _eval_sh0_idft(sh, timestamp, degree_t, max_degree_t):
    times = torch.as_tensor(timestamp, device="cuda", dtype=torch.float32).reshape(-1, 1)
    indices = torch.arange(degree_t + 1, device="cuda")
    basis = torch.empty((times.shape[0], degree_t + 1), device="cuda")
    basis[:, indices % 2 == 0] = torch.cos(torch.pi * times * indices[indices % 2 == 0])
    basis[:, indices % 2 == 1] = torch.sin(torch.pi * times * (indices[indices % 2 == 1] + 1))
    return torch.cat((torch.sum(sh[:, : degree_t + 1, :] * basis[0, :, None], dim=1, keepdim=True), sh[:, max_degree_t + 1 :, :]), dim=1)


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

    def _object_attributes(self, timestamp, camera_center, extra_bboxes):
        if isinstance(extra_bboxes, dict):
            valid_gaussians = [gaussian for gaussian in self.object_gaussians if gaussian.id in extra_bboxes]
        else:
            valid_gaussians = [gaussian for gaussian in self.object_gaussians if gaussian.bounding_box.first_timestamp <= timestamp <= gaussian.bounding_box.last_timestamp]

        rendered = valid_gaussians
        rigid = {
            "means3D": [torch.empty((0, 3), device="cuda")],
            "opacities": [torch.empty((0, 1), device="cuda")],
            "scales": [torch.empty((0, 3), device="cuda")],
            "rotations": [torch.empty((0, 4), device="cuda")],
            "shs": [torch.empty((0, (self.rigid_max_sh_degree + 1) ** 2 + self.rigid_max_sh_degree_t, 3), device="cuda")],
            "object_rotations": [torch.empty((0, 4), device="cuda")],
            "object_translations": [torch.empty((0, 3), device="cuda")],
        }
        nonrigid = {
            "means3D": [torch.empty((0, 3), device="cuda")],
            "opacities": [torch.empty((0, 1), device="cuda")],
            "scales": [torch.empty((0, 3), device="cuda")],
            "rotations": [torch.empty((0, 4), device="cuda")],
            "shs": [torch.empty((0, (self.nonrigid_max_sh_degree + 1) ** 2 * (self.nonrigid_max_sh_degree_t + 1), 3), device="cuda")],
            "ts": [torch.empty((0, 1), device="cuda")],
            "scale_ts": [torch.empty((0, 1), device="cuda")],
            "rotation_rs": [torch.empty((0, 4), device="cuda")],
            "object_rotations": [torch.empty((0, 4), device="cuda")],
            "object_translations": [torch.empty((0, 3), device="cuda")],
        }
        if isinstance(extra_bboxes, dict):
            rigid["object_rotations"] = [torch.empty((0, 3, 3), device="cuda")]
            nonrigid["object_rotations"] = [torch.empty((0, 3, 3), device="cuda")]
        for gaussian in rendered:
            if gaussian.amount == 0:
                continue
            target = rigid if gaussian.bounding_box.object_type == "rigid" else nonrigid
            target["means3D"].append(gaussian.get_xyz())
            target["opacities"].append(gaussian.get_opacity())
            target["scales"].append(gaussian.get_scaling())
            target["rotations"].append(gaussian.get_quaternion())
            target["shs"].append(gaussian.get_features())
            if gaussian.bounding_box.object_type == "nonrigid":
                target["ts"].append(gaussian.get_t())
                target["scale_ts"].append(gaussian.get_scaling_t())
                target["rotation_rs"].append(gaussian.get_quaternion_r())
            if isinstance(extra_bboxes, dict):
                target["object_rotations"].append(extra_bboxes[gaussian.id][:3, :3].expand(gaussian.amount, 3, 3))
                target["object_translations"].append(extra_bboxes[gaussian.id][:3, 3].expand(gaussian.amount, 3))
            else:
                target["object_rotations"].append(gaussian.bounding_box.get_quaternion(timestamp).expand(gaussian.amount, 4))
                target["object_translations"].append(gaussian.bounding_box.get_translation(timestamp).expand(gaussian.amount, 3))

        timestamp = (timestamp - self.timestamp_range[0]) / (self.timestamp_range[1] - self.timestamp_range[0])
        for key in rigid:
            rigid[key] = torch.cat(rigid[key], dim=0)
        rigid["cov3ds"] = compute_cov3d(rigid["scales"], rigid["rotations"])
        rigid["colors"] = compute_sh(
            active_degree=self.rigid_active_sh_degree,
            means3d=rigid["means3D"],
            shs=_eval_sh0_idft(rigid["shs"], timestamp, self.rigid_active_sh_degree_t, self.rigid_max_sh_degree_t),
            campos=camera_center,
        )

        for key in nonrigid:
            nonrigid[key] = torch.cat(nonrigid[key], dim=0)
        nonrigid["cov3ds"], covariance_t, speeds = compute_cov3d_conditional(nonrigid["scales"], nonrigid["scale_ts"], nonrigid["rotations"], nonrigid["rotation_rs"])
        delta_t = timestamp - nonrigid["ts"]
        nonrigid["opacities"] = nonrigid["opacities"] * torch.exp(-0.5 * delta_t * delta_t / covariance_t)
        nonrigid["means3D"] = nonrigid["means3D"] + delta_t * speeds
        nonrigid["colors"] = compute_sh_fourier(
            self.nonrigid_active_sh_degree,
            self.nonrigid_active_sh_degree_t,
            means3d=nonrigid["means3D"],
            mean_ts=nonrigid["ts"],
            shs=nonrigid["shs"],
            campos=camera_center,
            timestamp=timestamp,
            time_duration=1,
        )

        means = torch.cat((rigid["means3D"], nonrigid["means3D"]), dim=0)
        opacities = torch.cat((rigid["opacities"], nonrigid["opacities"]), dim=0)
        colors = torch.cat((rigid["colors"], nonrigid["colors"]), dim=0)
        covariances = torch.cat((rigid["cov3ds"], nonrigid["cov3ds"]), dim=0)
        rotations = torch.cat((rigid["object_rotations"], nonrigid["object_rotations"]), dim=0)
        translations = torch.cat((rigid["object_translations"], nonrigid["object_translations"]), dim=0)
        if rotations.shape[1] == 4:
            rotations = build_rotation(rotations)
        covariance_matrices = torch.zeros((covariances.shape[0], 3, 3), device="cuda")
        covariance_matrices[:, 0, 0] = covariances[:, 0]
        covariance_matrices[:, 0, 1] = covariance_matrices[:, 1, 0] = covariances[:, 1]
        covariance_matrices[:, 0, 2] = covariance_matrices[:, 2, 0] = covariances[:, 2]
        covariance_matrices[:, 1, 1] = covariances[:, 3]
        covariance_matrices[:, 1, 2] = covariance_matrices[:, 2, 1] = covariances[:, 4]
        covariance_matrices[:, 2, 2] = covariances[:, 5]
        covariance_matrices = rotations @ covariance_matrices @ rotations.transpose(1, 2)
        covariance_world = torch.stack((covariance_matrices[:, 0, 0], covariance_matrices[:, 0, 1], covariance_matrices[:, 0, 2], covariance_matrices[:, 1, 1], covariance_matrices[:, 1, 2], covariance_matrices[:, 2, 2]), dim=-1)
        return (rotations @ means.unsqueeze(-1)).squeeze(-1) + translations, opacities, covariance_world, colors

    def get_gaussian_attributes(self, timestamp, camera_center, full_proj_transform, visibility, extra_bboxes, **_):
        means = [torch.empty((0, 3), device="cuda")]
        opacities = [torch.empty((0, 1), device="cuda")]
        covariances = [torch.empty((0, 6), device="cuda")]
        colors = [torch.empty((0, 3), device="cuda")]
        if "background" in visibility:
            means.append(self.bkgd_gaussians.get_xyz())
            opacities.append(self.bkgd_gaussians.get_opacity())
            covariances.append(compute_cov3d(self.bkgd_gaussians.get_scaling(), self.bkgd_gaussians.get_quaternion()))
            colors.append(compute_sh(self.bkgd_active_sh_degree, self.bkgd_gaussians.get_xyz(), self.bkgd_gaussians.get_features(), camera_center))
        if "object" in visibility and self.is_dynamic:
            object_means, object_opacities, object_covariances, object_colors = self._object_attributes(timestamp, camera_center, extra_bboxes)
            means.append(object_means)
            opacities.append(object_opacities)
            covariances.append(object_covariances)
            colors.append(object_colors)
        return {"means3D": torch.cat(means), "opacities": torch.cat(opacities), "cov3ds": torch.cat(covariances), "colors": torch.cat(colors)}

    @torch.no_grad()
    def render_batch(self, timestamp, camera_params, ray_d, extra_bboxes):
        attributes = [self.get_gaussian_attributes(timestamp=timestamp, visibility=("background", "object", "skymap"), extra_bboxes=extra_bboxes, **camera) for camera in camera_params]
        rasterizer = BatchedGaussianRasterizer(
            [_raster_settings(camera, self.background) for camera in camera_params],
            init_buffer_size=len(attributes[0]["means3D"]),
            init_texture_size=[camera_params[0]["image_height"], camera_params[0]["image_width"]],
        )
        rgb, alpha = rasterizer(
            means3D=attributes[0]["means3D"],
            means2D=None,
            opacities=attributes[0]["opacities"],
            colors_precomp=torch.stack([attribute["colors"] for attribute in attributes]),
            cov3D_precomp=attributes[0]["cov3ds"],
        )
        if hasattr(self, "sky_cubemap"):
            rgb = torch.lerp(self.sky_cubemap(ray_d), rgb, alpha)
        return rgb.detach().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
