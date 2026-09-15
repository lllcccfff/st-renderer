import math

import torch


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


def gen_rays(R, T, K, height, width):
    rays_o = -torch.matmul(R, T).squeeze()
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=R.device),
        torch.arange(height, dtype=torch.float32, device=R.device),
        indexing="xy",
    )
    pixels = torch.stack([i + 0.5, j + 0.5, torch.ones_like(i)], dim=2)
    rays_d = torch.matmul(pixels, torch.inverse(K).T)
    rays_d = torch.matmul(rays_d, R.T)
    rays_d = rays_d / torch.norm(rays_d, dim=2, keepdim=True)
    return rays_o.expand_as(rays_d), rays_d
