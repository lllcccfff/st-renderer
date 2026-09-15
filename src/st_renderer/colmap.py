from __future__ import annotations

import collections
import struct
from pathlib import Path

import numpy as np


Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name"])

_CAMERA_MODEL_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_next(file, size, fmt):
    return struct.unpack("<" + fmt, file.read(size))


def _read_cameras(path: Path):
    cameras = {}
    with path.open("rb") as file:
        for _ in range(_read_next(file, 8, "Q")[0]):
            camera_id, model_id, width, height = _read_next(file, 24, "iiQQ")
            model, count = _CAMERA_MODEL_PARAMS[model_id]
            params = np.array(_read_next(file, 8 * count, "d" * count))
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def _read_images(path: Path):
    images = {}
    with path.open("rb") as file:
        for _ in range(_read_next(file, 8, "Q")[0]):
            values = _read_next(file, 64, "idddddddi")
            image_id = values[0]
            qvec = np.array(values[1:5])
            tvec = np.array(values[5:8])
            camera_id = values[8]
            name = bytearray()
            while (character := file.read(1)) != b"\x00":
                name.extend(character)
            points = _read_next(file, 8, "Q")[0]
            file.seek(24 * points, 1)
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name.decode())
    return images


def read_colmap(path: Path):
    return _read_cameras(path / "cameras.bin"), _read_images(path / "images.bin")


def qvec2rotmat(qvec):
    return np.array(
        [
            [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
        ]
    )


def image_index(images):
    indexed = {}
    for image in images.values():
        camera_name, timestamp = image.name.rsplit(".", 1)[0].rsplit("_", 1)
        indexed.setdefault(camera_name, {})[int(timestamp)] = image
    return indexed
