# FastGauss fused multi-camera pipeline plan

## Scope

This plan moves StreetGaussianScene's per-frame Gaussian preparation into the OpenGL-based FastGauss batch pipeline. It applies only to `submodules/st-renderer` and FastGauss. It does not modify EasyDrive, NSG, or the visualizer.

The target is six synchronized cameras. `get_gaussian_attributes` is called once per frame and never performs object frustum filtering.

The intended batch layout follows gsplat: scene tensors have a Gaussian dimension and camera tensors add a camera dimension. The CUDA pipeline processes camera/Gaussian pairs instead of invoking Python once per camera. See the [gsplat batch contract](https://github.com/nerfstudio-project/gsplat/blob/main/docs/batch.md).

## Definitions

Object filtering and raster work scheduling are distinct:

- `get_gaussian_attributes` selects objects only from timestamp and `current_objects`; no camera projection or frustum predicate is allowed there.
- FastGauss may mark a `(camera, gaussian)` pair as non-drawable while preparing that camera's VBO. This does not remove the object from the source set.

The existing `prefiltered=True` change is incorrect for this requirement and must be reverted before implementation.

## Per-frame data flow

```text
StreetGaussianScene.get_gaussian_attributes(timestamp, current_objects)  # once
    raw Gaussian buffers
    gid_map[gid] = (type, object_id)
    object pose table
                         |
                         v
FastGauss CUDA world_prepare[N]
    gid identifies background / rigid / nonrigid
    object transform, covariance, temporal opacity and motion
                         |
                         v
FastGauss CUDA view_prepare[C, N]
    camera_id, gid identify the camera/Gaussian pair
    view-dependent SH, projection, depth and draw visibility
                         |
                         v
GPU compact + segmented depth sort + VBO pack
                         |
                         v
one OpenGL layered multi-draw for C camera layers
```

## Data contract

`get_gaussian_attributes` becomes a one-call source builder. It accepts timestamp and object poses, but no camera center or projection matrix. Its output contains raw GPU tensors, `gid_map`, and object pose metadata.

At load time, Gaussian storage is finalized once:

- scaling uses `exp`;
- opacity uses `sigmoid`;
- quaternions are normalized;
- DC and remaining SH coefficients are concatenated.

All Gaussian getters then return stored tensors directly.

`gid_map` is a GPU tensor with one `(type, object_id)` pair per Gaussian. Background entries use an invalid object id. CUDA dispatch is exactly:

```cpp
const GaussianMap map = gid_map[gid];
const int type = map.type;
const int object_id = map.object_id;
```

Geometry buffers are indexed directly by `gid`; no class segment boundary or class count participates in dispatch. SH buffers stay type-specific, so the background is never padded to a dynamic object's wider SH layout. `object_sources[object_id] = (global_gid_start, type_feature_start)` converts an object's global `gid` to its row in the selected type SH buffer; it is an object-id lookup, not a dispatch boundary. Per-frame object transforms are stored once per active object and looked up through `object_id`, rather than expanded in Python for every Gaussian.

## Fused CUDA stages

### 1. World stage: `world_prepare<<<N>>>`

One thread owns one Gaussian. It loads `(type, object_id)` from `gid_map[gid]` and performs all camera-independent work:

- background: covariance;
- rigid: covariance, temporal SH preparation, object rotation and translation;
- nonrigid: conditional covariance, temporal mean displacement, temporal opacity, object rotation and translation.

This replaces Python/MixPlat calls to `compute_cov3d`, `compute_cov3d_conditional`, per-Gaussian object transforms, and associated tensor assembly. The formulas must match the current MixPlat CUDA implementation exactly.

### 2. View stage: `view_prepare<<<C * N>>>`

For linear thread index `tid`:

```cpp
camera_id = tid / N;
gid = tid % N;
```

The stage reads world-stage output, evaluates class-specific view-dependent SH using that camera center, and emits depth plus a draw-visibility flag. Background, rigid, and nonrigid use their current SH/SH-Fourier formulas, fused here rather than invoked from Python.

Covariance and object transforms are deliberately not recomputed for each camera.

## FastGauss batch rendering

The current FastGauss batch path loops in Python per camera for frustum filtering, gathering, depth sorting, and VBO packing. Replace that preprocessing with GPU work:

1. compact drawable `(camera, gaussian)` pairs;
2. sort each camera's compacted range back-to-front by depth;
3. pack the sorted records directly into the CUDA-mapped OpenGL VBO;
4. write GPU-side indirect draw commands;
5. issue one layered multi-draw using the existing draw-id shader path.

This removes six independent CUDA/host preparation paths. CUDA work remains on the current stream; CUDA-GL map/unmap is the only required graphics boundary. There is no serial rendering fallback.

The implementation must first verify EGL support for the required indirect multi-draw path. If unavailable, it fails immediately instead of silently switching to per-camera draws.

## Files expected to change

- `submodules/st-renderer/src/st_renderer/gaussians.py`: load-time activation and direct getters.
- `submodules/st-renderer/src/st_renderer/scene.py`: one source-builder invocation per frame and raw FastGauss input.
- `fast-gauss-paral/fast_gauss/gsplat_utils.py`: fused CUDA preprocessing, GPU sort/pack, and indirect draw integration.
- `fast-gauss-paral/fast_gauss/__init__.py`: the direct batch API for raw class buffers.
- FastGauss batch shaders only if indirect draw metadata requires a shader-side change.

## Validation

1. Freeze a six-camera scene, timestamp, intrinsics, extrinsics, and object poses.
2. Compare fused world-stage output with the existing MixPlat path for each class: means, covariance, opacity, and colors.
3. Render all six cameras with the old path and fused path; retain output images and per-camera pixel diffs.
4. Verify an active object outside every camera frustum remains in the source candidate set.
5. Profile after warmup with CUDA events at source building, world stage, view stage, compact/sort, VBO pack, GL draw, and readback.
6. Report observer total latency plus p50/p95 and per-stage timings against the current baseline.

No package or environment installation is part of this work.
