from __future__ import annotations

import torch
import numpy as np
from plyfile import PlyData
from torch import nn


def _tensor(value):
    return value.detach().to(device="cuda", dtype=torch.float32).contiguous()


class GaussianModel:
    def __init__(self, max_sh_degree):
        self.max_sh_degree = max_sh_degree

    @property
    def amount(self):
        return self._xyz.shape[0]

    def get_scaling(self):
        return torch.exp(self._scaling)

    def get_quaternion(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    def get_xyz(self):
        return self._xyz

    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def load_ply(self, path):
        ply = PlyData.read(path)
        vertex = ply.elements[0]
        xyz = np.stack((np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])), axis=1)
        opacity = np.asarray(vertex["opacity"])[..., None]
        feature_dc = np.stack([np.asarray(vertex[f"f_dc_{index}"]) for index in range(3)], axis=1)[..., None]
        rest_names = sorted((item.name for item in vertex.properties if item.name.startswith("f_rest_")), key=lambda item: int(item.rsplit("_", 1)[1]))
        feature_rest = np.stack([np.asarray(vertex[name]) for name in rest_names], axis=1).reshape(len(vertex), 3, (self.max_sh_degree + 1) ** 2 - 1)
        scale_names = sorted((item.name for item in vertex.properties if item.name.startswith("scale_")), key=lambda item: int(item.rsplit("_", 1)[1]))
        rotation_names = sorted((item.name for item in vertex.properties if item.name.startswith("rot")), key=lambda item: int(item.rsplit("_", 1)[1]))
        self._xyz = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        self._features_dc = torch.tensor(feature_dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(feature_rest, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacity, dtype=torch.float32, device="cuda")
        self._scaling = torch.tensor(np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1), dtype=torch.float32, device="cuda")
        self._rotation = torch.tensor(np.stack([np.asarray(vertex[name]) for name in rotation_names], axis=1), dtype=torch.float32, device="cuda")


class GaussianBackground(GaussianModel):
    pass


class GaussianObject(GaussianModel):
    def __init__(self, bounding_box, max_sh_degree, max_sh_degree_t):
        super().__init__(max_sh_degree)
        self.bounding_box = bounding_box.cuda()
        self.max_sh_degree_t = max_sh_degree_t

    @property
    def id(self):
        return self.bounding_box.id

    def load(self, state):
        self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, self._semantic, self.max_radii2d, self.xyz_gradient_accum, self.denom, _, self.extent = state
        self._xyz = _tensor(self._xyz)
        self._features_dc = _tensor(self._features_dc)
        self._features_rest = _tensor(self._features_rest)
        self._scaling = _tensor(self._scaling)
        self._rotation = _tensor(self._rotation)
        self._opacity = _tensor(self._opacity)


class GaussianModelRigid(GaussianObject):
    pass


class GaussianModelNonrigid(GaussianObject):
    def load(self, state):
        self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, self._t, self._scaling_t, self._rotation_r, self._semantic, self.max_radii2d, self.xyz_gradient_accum, self.t_gradient_accum, self.denom, _, self.extent = state
        self._xyz = _tensor(self._xyz)
        self._features_dc = _tensor(self._features_dc)
        self._features_rest = _tensor(self._features_rest)
        self._scaling = _tensor(self._scaling)
        self._rotation = _tensor(self._rotation)
        self._opacity = _tensor(self._opacity)
        self._t = _tensor(self._t)
        self._scaling_t = _tensor(self._scaling_t)
        self._rotation_r = _tensor(self._rotation_r)

    def get_t(self):
        return self._t

    def get_scaling_t(self):
        return torch.exp(self._scaling_t)

    def get_quaternion_r(self):
        return torch.nn.functional.normalize(self._rotation_r, dim=-1)


class SkyCubeMap(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.sky_cube_map = nn.Parameter(torch.full((6, resolution, resolution, 3), 0.8, device="cuda"), requires_grad=False)

    def load(self, state):
        self.load_state_dict(state["params"])

    def forward(self, ray_d):
        import nvdiffrast.torch as dr

        return dr.texture(self.sky_cube_map[None], ray_d, filter_mode="linear", boundary_mode="cube").permute(0, 3, 1, 2).clamp(0, 1)
