# ST Renderer

Standalone StreetWorld renderer for trained Street Gaussian scenes. It has no
EasyDrive import and exposes the interface StreetWorld already uses:

```python
from st_renderer import SimulatorInterface

renderer = SimulatorInterface()
renderer.load_metadata(scene_yaml)
renderer.load_model(scene_yaml)
renderer.update_scene(timestamp, object_poses)
images = renderer.render(K, heights, widths, extrinsics)
```

`K` is `[B, 3, 3]`, `extrinsics` is `[B, 4, 4]`, and `heights`/`widths` are
Python lists. `render()` returns uint8 RGB `[B, H, W, 3]`.

The scene YAML remains the source of scene data and checkpoints. Relative
paths are resolved from the directory that contains its `configs/` tree.
Runtime requirements outside this repository are the local `fast_gauss`,
`mixplat`, and `nvdiffrast` CUDA packages; `trajdata` is only used for
NuScenes maps.
