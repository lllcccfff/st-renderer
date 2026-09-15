# Simulator Interface

Independent batched Fast Gauss rendering runtime. It has no EasyDrive import.

The caller supplies a scene loader with two methods:

```python
loader.load_metadata(scene_id)
loader.load_model(scene_id)  # returns a scene model
```

The returned scene model must provide `background`,
`get_gaussian_attributes(...)`, and optionally `sky_cubemap(ray_d)`.

```python
from simulator_interface import SimulatorInterface

simulator = SimulatorInterface(loader)
metadata = simulator.load_metadata(scene_id)
simulator.load_model(scene_id)
simulator.update_scene(timestamp, object_poses)
images = simulator.render(K, heights, widths, extrinsics)
```

`K` has shape `[B, 3, 3]`, `extrinsics` has shape `[B, 4, 4]`, and
`heights`/`widths` are Python lists. `render()` returns a uint8 ndarray with
shape `[B, H, W, 3]`. CPU tensors are pinned and uploaded asynchronously;
CUDA tensors are used directly.

The runtime requires PyTorch and the local `fast_gauss` package from
[`fast-gauss-paral`](../fast-gauss-paral).
