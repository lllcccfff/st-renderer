# ST Renderer

Standalone StreetWorld renderer for trained Street Gaussian scenes. It has no
EasyDrive import and exposes the interface StreetWorld already uses:

```python
from st_renderer import SimulatorInterface

renderer = SimulatorInterface("nuscenes")
renderer.load_metadata(scene_name)
renderer.load_model(scene_name)
renderer.update_scene(timestamp, object_poses)
images = renderer.render(K, heights, widths, extrinsics)
renderer.close()
```

`K` is `[B, 3, 3]`, `extrinsics` is `[B, 4, 4]`, and `heights`/`widths` are
Python lists. `render()` returns uint8 RGB `[B, H, W, 3]`.

Fast Gauss initializes OpenGL on the first render. Run all rendering and
`close()` on the same thread; importing the renderer does not create an OpenGL
context. Reuse that thread across scene resets so the context and render buffers
can be reused. `close()` releases Fast Gauss resources and its owned EGL context.

`scene_name` is a four-digit NuScenes name or a Waymo name such as
`001-segment-1422926405879888210`. Scene metadata and model data are loaded
from one NPZ archive under the dataset root. NuScenes vector maps are loaded
from `map_cache.npz` under the same root. Waymo ground meshes are loaded from
`ground/<scene_name>.obj` under that root.

Runtime requirements outside this repository are the local `fast_gauss`,
`mixplat`, and `nvdiffrast` CUDA packages; `trajdata` is only used for
NuScenes maps.
