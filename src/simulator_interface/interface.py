import torch

from .camera import gen_rays, get_fov, get_projection_matrix
from .renderer import BatchedFastGaussRenderer


def _cuda_async(tensor):
    return tensor if tensor.is_cuda else tensor.pin_memory().cuda(non_blocking=True)


class SimulatorInterface:
    def __init__(self, scene_loader, z_near=0.0001, z_far=1000):
        self.scene_loader = scene_loader
        self.z_near = z_near
        self.z_far = z_far
        self.current_timestamp = None
        self.current_objects = {}
        self.renderer = BatchedFastGaussRenderer()

    def load_metadata(self, scene_id):
        return self.scene_loader.load_metadata(scene_id)

    def load_model(self, scene_id):
        self.scene_model = self.scene_loader.load_model(scene_id)
        return self.scene_model

    def update_scene(self, timestamp, object_poses):
        self.current_timestamp = timestamp
        self.current_objects = {object_id: _cuda_async(pose) for object_id, pose in object_poses.items()}

    def render(self, K, H, W, extrinsics):
        fovs = [get_fov(camera_K, height, width) for camera_K, height, width in zip(K, H, W)]
        projections = [
            get_projection_matrix(camera_K, height, width, self.z_near, self.z_far)
            for camera_K, height, width in zip(K, H, W)
        ]
        K = _cuda_async(K)
        w2c = _cuda_async(extrinsics).contiguous()

        camera_params = []
        ray_d = []
        for camera_K, camera_w2c, height, width, (fov_x, fov_y), projection in zip(
            K, w2c, H, W, fovs, projections
        ):
            projection = _cuda_async(projection)
            camera_params.append(
                {
                    "world_view_transform": camera_w2c.T,
                    "full_proj_transform": (projection @ camera_w2c).contiguous().T,
                    "camera_center": camera_w2c.inverse()[:3, 3].contiguous(),
                    "image_height": height,
                    "image_width": width,
                    "FoVx": fov_x,
                    "FoVy": fov_y,
                }
            )
            ray_d.append(gen_rays(camera_w2c[:3, :3].T, camera_w2c[:3, 3], camera_K, height, width)[1])
        return self.renderer.render(
            self.scene_model,
            self.current_timestamp,
            camera_params,
            torch.stack(ray_d),
            self.current_objects,
        )
