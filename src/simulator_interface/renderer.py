import math

import torch

from fast_gauss import BatchedGaussianRasterizer, GaussianRasterizationSettings, GaussianRasterizer


def raster_settings(camera_params, background):
    return GaussianRasterizationSettings(
        image_height=camera_params["image_height"],
        image_width=camera_params["image_width"],
        tanfovx=math.tan(camera_params["FoVx"] * 0.5),
        tanfovy=math.tan(camera_params["FoVy"] * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=camera_params["world_view_transform"],
        projmatrix=camera_params["full_proj_transform"],
        campos=camera_params["camera_center"],
        prefiltered=False,
    )


class FastGaussRenderer:
    def __call__(self, camera_params, gaussian_attributes, background):
        rgb, alpha = GaussianRasterizer(raster_settings(camera_params, background))(
            means3D=gaussian_attributes["means3D"],
            means2D=None,
            opacities=gaussian_attributes["opacities"],
            colors_precomp=gaussian_attributes["colors"],
            cov3D_precomp=gaussian_attributes["cov3ds"],
        )
        return {"rgb": rgb, "alpha": alpha}


class BatchedFastGaussRenderer:
    @torch.no_grad()
    def render(self, scene, timestamp, camera_params, ray_d, object_poses):
        image_size = (camera_params[0]["image_height"], camera_params[0]["image_width"])
        gaussian_attributes = [
            scene.get_gaussian_attributes(
                **params,
                timestamp=timestamp,
                visibility=["background", "object", "skymap"],
                extra_bboxes=object_poses,
            )
            for params in camera_params
        ]
        rasterizer = BatchedGaussianRasterizer(
            [raster_settings(params, scene.background) for params in camera_params],
            init_buffer_size=len(gaussian_attributes[0]["means3D"]),
            init_texture_size=list(image_size),
        )
        rgb, alpha = rasterizer(
            means3D=gaussian_attributes[0]["means3D"],
            means2D=None,
            opacities=gaussian_attributes[0]["opacities"],
            colors_precomp=torch.stack([attrs["colors"] for attrs in gaussian_attributes]),
            cov3D_precomp=gaussian_attributes[0]["cov3ds"],
        )
        if hasattr(scene, "sky_cubemap"):
            rgb = torch.lerp(scene.sky_cubemap(ray_d), rgb, alpha)
        return rgb.detach().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
