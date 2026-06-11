""" Image loading, resizeing, and normalisation transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from depth_estimation.config import settings
from depth_estimation.utils import get_logger

log  = get_logger(__name__)

RawInput = Union[np.ndarray, Image.Image, str, Path]

class ImagePreProcessor:

    def __init__(self, max_dimension: int | None = None, from_bgr: bool = False) -> None:

        self.max_dimension = max_dimension
        self.from_bgr = from_bgr # set true when the input arrays come from cv2.VideoCapture


    # ----- Public API ---------------------------------------------------------------------

    def load(self, source: RawInput) -> np.ndarray:
        """ Return a uint8 RGB numpy array from any supported source."""
        image = self._to_rgb_array(source)
        if self.from_bgr and isinstance(source, np.ndarray):
            image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        if self.max_dimension:
            image = self._resize(image, self.max_dimension)
        return image
    

    def load_pil(self, source: RawInput) -> Image.Image:
        """ Return a PIL image (RGB) from any supported source."""
        return Image.fromarray(self.load(source))
    
    # ----- Helpers -----------------------------------------------------------------

    @staticmethod
    def _to_rgb_array(source: RawInput) -> np.ndarray:
        if isinstance(source, (str,Path)):
            img = Image.open(source).convert("RGB")
            return np.array(img, dtype=np.uint8)
        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"), dtype=np.uint8)
        if isinstance(source, np.ndarray):
            arr = source
            # if image is gray color
            if arr.ndim == 2:
                # Convert it into 3D
                arr = np.stack([arr]*3, axis=-1)
            # if image is RGBA (four channels)
            elif arr.shape[2] == 4:
                # remove alpha layer
                arr = arr[:, :, :3]
            return arr.astype(np.uint8)
        raise TypeError(f"Unsupported image type: {type(source)}")
    
    @staticmethod
    def _resize(image: np.ndarray, max_dim: int) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h,w) <= max_dim:
            return image
        scale = max_dim / max(h,w)
        new_w = max(1,int(w*scale))
        new_h = max(1,int(h*scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        log.debug("Pre-resize %dx%d -> %dx%d", w, h, new_w, new_h)
        return resized


# ----- Webcam helpers ------------------------------------------------------------

@staticmethod
def open_webcam( device_id: int = 0, widht: int = 640, height: int = 640) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, widht)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Canot open webcam device {device_id}")
    log.info(
        "Webcam %d opended at %dx%d", device_id,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return cap

@staticmethod
def read_frame_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    """read one frame from the webcam."""
    ok, frame = cap.read()
    if not ok:
        return None
    # Open CV gives images in BGR.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

