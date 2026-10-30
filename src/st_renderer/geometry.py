from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as functional


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_shape + (9,)), dim=-1)
    q_abs = torch.sqrt(torch.clamp(torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), dim=-1), min=0))
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2 * q_abs[..., None].clamp_min(0.1))
    return candidates[functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5].reshape(batch_shape + (4,))


def quaternion_multiply(left, right):
    lw, lx, ly, lz = torch.unbind(left, -1)
    rw, rx, ry, rz = torch.unbind(right, -1)
    return torch.stack((lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw), -1)


def build_rotation(quaternion):
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    rotation = torch.empty((*quaternion.shape[:-1], 3, 3), device=quaternion.device, dtype=quaternion.dtype)
    rotation[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[..., 0, 1] = 2 * (x * y - w * z)
    rotation[..., 0, 2] = 2 * (x * z + w * y)
    rotation[..., 1, 0] = 2 * (x * y + w * z)
    rotation[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[..., 1, 2] = 2 * (y * z - w * x)
    rotation[..., 2, 0] = 2 * (x * z - w * y)
    rotation[..., 2, 1] = 2 * (y * z + w * x)
    rotation[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


def slerp(first, second, ratio):
    first = first / torch.norm(first, dim=-1, keepdim=True)
    second = second / torch.norm(second, dim=-1, keepdim=True)
    dot = (first * second).sum(-1)
    if dot < 0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        return functional.normalize((1 - ratio) * first + ratio * second, dim=-1)
    theta = torch.acos(dot)
    return (torch.sin(theta * (1 - ratio)) / torch.sin(theta)) * first + (torch.sin(theta * ratio) / torch.sin(theta)) * second


class Trajectory:
    def __init__(self, values: dict[int, Iterable] | None = None):
        self.trans = {}
        self.quats = {}
        self._sorted_ts = []
        if values:
            for timestamp, transform in values.items():
                self._add(timestamp, transform)

    def __len__(self):
        return len(self._sorted_ts)

    def _add(self, timestamp, transform):
        timestamp = int(timestamp)
        if timestamp not in self._sorted_ts:
            transform = torch.as_tensor(transform, dtype=torch.float32)
            self.trans[timestamp] = transform[:3, 3]
            self.quats[timestamp] = matrix_to_quaternion(transform[:3, :3])
            self._sorted_ts.append(timestamp)
            self._sorted_ts.sort()

    def _sample(self, timestamp):
        timestamp = int(timestamp)
        if not self._sorted_ts:
            raise ValueError("Empty trajectory.")
        if timestamp < self._sorted_ts[0] or timestamp > self._sorted_ts[-1]:
            raise ValueError(f"Timestamp {timestamp} is outside trajectory.")
        for first, second in zip(self._sorted_ts[:-1], self._sorted_ts[1:]):
            if first <= timestamp <= second:
                return first, second, (timestamp - first) / (second - first)
        raise ValueError(f"Timestamp {timestamp} could not be indexed.")

    def get_quaternion(self, timestamp):
        first, second, ratio = self._sample(timestamp)
        if ratio == 0:
            return self.quats[first]
        if ratio == 1:
            return self.quats[second]
        return slerp(self.quats[first], self.quats[second], ratio)

    def get_translation(self, timestamp):
        first, second, ratio = self._sample(timestamp)
        if ratio == 0:
            return self.trans[first]
        if ratio == 1:
            return self.trans[second]
        return (1 - ratio) * self.trans[first] + ratio * self.trans[second]

    def get_rotation(self, timestamp):
        return build_rotation(self.get_quaternion(timestamp))

    def get_transform(self, timestamp):
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = self.get_rotation(timestamp)
        transform[:3, 3] = self.get_translation(timestamp)
        return transform


class BoundingBox(Trajectory):
    def __init__(self, identifier, detail_type, size):
        super().__init__()
        self.id = identifier
        self.detail_type = detail_type
        self.size = torch.as_tensor(size, dtype=torch.float32)

    @property
    def object_type(self):
        return {"vehicle": "rigid", "pedestrian": "nonrigid", "cyclist": "nonrigid", "rigid": "rigid", "nonrigid": "nonrigid"}[self.detail_type]

    @property
    def first_timestamp(self):
        return self._sorted_ts[0]

    @property
    def last_timestamp(self):
        return self._sorted_ts[-1]

    def get_sorted_ts(self):
        return self._sorted_ts

    def add_timestamp(self, timestamp, transform):
        self._add(timestamp, transform)

    def cuda(self):
        self.size = self.size.cuda()
        self.trans = {timestamp: value.cuda() for timestamp, value in self.trans.items()}
        self.quats = {timestamp: value.cuda() for timestamp, value in self.quats.items()}
        return self

    def import_offset(self, offsets):
        for timestamp in self._sorted_ts:
            offset = torch.as_tensor(offsets[str(timestamp)], dtype=torch.float32, device=self.size.device)
            self.quats[timestamp] = quaternion_multiply(matrix_to_quaternion(offset[:3, :3]), self.quats[timestamp])
            self.trans[timestamp] = offset[:3, 3] + self.trans[timestamp]


class Camera:
    def __init__(self, name, height, width, K):
        self.name = name
        self.image_height = height
        self.image_width = width
        self.K = K.cuda().contiguous()
        self.w2c = {}
        self.world_view_transform = {}

    def add_ext(self, timestamp, w2c):
        w2c = w2c.cuda().contiguous()
        self.w2c[timestamp] = w2c
        self.world_view_transform[timestamp] = w2c.T

    def import_offset(self, offsets):
        for timestamp, w2c in self.w2c.items():
            offset = torch.as_tensor(offsets[str(timestamp)], dtype=w2c.dtype, device=w2c.device)
            self.w2c[timestamp] = (w2c.inverse() @ offset).inverse()
            self.world_view_transform[timestamp] = self.w2c[timestamp].T


def get_fov(K, height, width):
    return 2 * math.atan(width / 2 / K[0, 0]), 2 * math.atan(height / 2 / K[1, 1])


def get_projection_matrix(K, height, width, z_near, z_far):
    projection = torch.zeros((4, 4), dtype=K.dtype, device=K.device)
    projection[0, 0] = 2 * K[0, 0] / width
    projection[0, 1] = 2 * K[0, 1] / width
    projection[0, 2] = -1 + 2 * K[0, 2] / width
    projection[1, 1] = 2 * K[1, 1] / height
    projection[1, 2] = -1 + 2 * K[1, 2] / height
    projection[2, 2] = (z_far + z_near) / (z_far - z_near)
    projection[2, 3] = -2 * z_far * z_near / (z_far - z_near)
    projection[3, 2] = 1
    return projection


def gen_rays(rotation, translation, K, height, width):
    rays_o = -torch.matmul(rotation, translation).squeeze()
    x, y = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=rotation.device),
        torch.arange(height, dtype=torch.float32, device=rotation.device),
        indexing="xy",
    )
    pixels = torch.stack((x + 0.5, y + 0.5, torch.ones_like(x)), dim=2)
    rays_d = torch.matmul(torch.matmul(pixels, torch.inverse(K).T), rotation.T)
    rays_d = rays_d / torch.norm(rays_d, dim=2, keepdim=True)
    return rays_o.expand_as(rays_d), rays_d
