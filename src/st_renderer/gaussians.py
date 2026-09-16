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
        return self._scaling

    def get_quaternion(self):
        return self._rotation

    def get_xyz(self):
        return self._xyz

    def get_features(self):
        return self._features

    def get_opacity(self):
        return self._opacity

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
        self._features = torch.cat((
            torch.tensor(feature_dc, dtype=torch.float32, device="cuda").transpose(1, 2),
            torch.tensor(feature_rest, dtype=torch.float32, device="cuda").transpose(1, 2),
        ), dim=1)
        self._opacity = torch.sigmoid(torch.tensor(opacity, dtype=torch.float32, device="cuda"))
        self._scaling = torch.exp(torch.tensor(np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1), dtype=torch.float32, device="cuda"))
        self._rotation = torch.nn.functional.normalize(torch.tensor(np.stack([np.asarray(vertex[name]) for name in rotation_names], axis=1), dtype=torch.float32, device="cuda"), dim=-1)


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
        xyz, feature_dc, feature_rest, scaling, rotation, opacity, self._semantic, self.max_radii2d, self.xyz_gradient_accum, self.denom, _, self.extent = state
        self._xyz = _tensor(xyz)
        self._features = torch.cat((_tensor(feature_dc), _tensor(feature_rest)), dim=1)
        self._scaling = torch.exp(_tensor(scaling))
        self._rotation = torch.nn.functional.normalize(_tensor(rotation), dim=-1)
        self._opacity = torch.sigmoid(_tensor(opacity))


class GaussianModelRigid(GaussianObject):
    pass


class GaussianModelNonrigid(GaussianObject):
    def load(self, state):
        xyz, feature_dc, feature_rest, scaling, rotation, opacity, t, scaling_t, rotation_r, self._semantic, self.max_radii2d, self.xyz_gradient_accum, self.t_gradient_accum, self.denom, _, self.extent = state
        self._xyz = _tensor(xyz)
        self._features = torch.cat((_tensor(feature_dc), _tensor(feature_rest)), dim=1)
        self._scaling = torch.exp(_tensor(scaling))
        self._rotation = torch.nn.functional.normalize(_tensor(rotation), dim=-1)
        self._opacity = torch.sigmoid(_tensor(opacity))
        self._t = _tensor(t)
        self._scaling_t = torch.exp(_tensor(scaling_t))
        self._rotation_r = torch.nn.functional.normalize(_tensor(rotation_r), dim=-1)

    def get_t(self):
        return self._t

    def get_scaling_t(self):
        return self._scaling_t

    def get_quaternion_r(self):
        return self._rotation_r


class SkyCubeMap(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.sky_cube_map = nn.Parameter(torch.full((6, resolution, resolution, 3), 0.8, device="cuda"), requires_grad=False)

    def load(self, state):
        self.load_state_dict(state["params"])

    def forward(self, ray_d):
        import nvdiffrast.torch as dr

        return dr.texture(self.sky_cube_map[None], ray_d, filter_mode="linear", boundary_mode="cube").permute(0, 3, 1, 2).clamp(0, 1)
