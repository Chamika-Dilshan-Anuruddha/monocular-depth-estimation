"""Post-processing: colouries depth mps, blend with original, export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import matplotlib.cm as cm
import numpy as np
from PIL import Image

from depth_estimation.config import settings, ColormapName
from depth_estimation.utils import get_logger

log = get_logger(__name__)

_CMAPS = dict[str, matplotlib.colors.Colormap] = {
    name.value: cm.get_cmap(name.value) for name in ColormapName
}

class DepthPostprocessor:

    def __init__(self, colormap: ColormapName | str = ColormapName.INFRNO, invert: bool = True, alpha: float = 1.0) -> None:
        cmap_key = colormap.value if isinstance(colormap, ColormapName) else colormap
        if cmap_key not in _CMAPS:
            raise ValueError(
                f"Unknown colormap '{cmap_key}.",
                f"Chose from : {list(_CMAPS.keys())}"
            )
        self.cmap = _CMAPS[cmap_key]
        self.invert = invert
        self.alpha = alpha
        self.colormap_name = cmap_key

    # ----- Primary output --------------------------------------------------------------------

    def colourise(self, depth:np.ndarray) -> np.ndarray:
        """Map a H x W float 32 depth array -> H x W x 3"""
        depth = self._validate(depth)
        if self.invert:
            depth = 1.0 - depth  #  near objects appear brighter
        coloured = (self.cmap(depth)[:, :, :3]*255).astype(np.uint8)  # depth values are range from 0 to 1. cmap gives RGBA image.
        return coloured
    
    def blend(self, depth: np.ndarray, original: np.ndarray) -> np.ndarray:
        """ Blend colourised depth with the original RGB image"""
        coloured = self.colourise(depth)

        if coloured.shape[:2] != original.shape[:2]:
            h,w = original.shape[:2]
            coloured = cv2.resize(coloured, (w,h), interpolation=cv2.INTER_LINEAR)

        return cv2.addWeighted(
            coloured, self.alpha,
            original.astype(np.uint8), 1.0 - self.alpha,
            0  # formula is output = alpha*coloured + (1-alpha)*original
        )
    
    def side_by_side(self, depth: np.ndarray, original: np.ndarray, gap: int = 8, gap_colour: tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
        """Return original and depth map laid out side-by-side. output looks : original image ::: some gap::: Depth image"""

        coloured = self.colourise(depth)
        orig = original.astype(np.uint8)

        h = max(orig.shape[0], coloured.shape[0])
        if orig.shape[0]  != h:
            orig = cv2.resize(orig, (orig.shape[1], h))
        if coloured.shape[0]  != h:
            coloured = cv2.resize(coloured, (coloured.shape[1], h))

        divider = np.full((h, gap, 3), gap_colour, dtype=np.uint8)
        return np.concatenate([orig, divider, coloured], axis=1)
    

    # ---- Metrics helpers -------------------------------------------------------------

    @staticmethod
    def depth_stats(depth: np.ndarray) -> dict[str, float]:
        """ Returns basic statistics of a depth map (for UI display)."""
        return {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
            "std": float(depth.std())
        }
    
    # ----- Export --------------------------------------------------------------

    def save(self, depth: np.ndarray, path: str | Path, original: Optional[np.ndarray] = None, mode: str = "colour") -> Path:
        """Save the visualisation to *path*."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "raw":
            depth16 = (depth * 65535).astype(np.uint16) # 16 bit iamge can store values from 0 to 65535
            Image.fromarray(depth16, mode="I;16").save(out_path) # I means Interger image, 16 means 16 bits per pixel
        elif mode == "blend":
            assert original is not None, "original required for blend mode" # This message is shown when the condigion is False
            img = self.blend(depth, original)
            Image.fromarray(img).save(out_path)
        elif mode == "side_by_side":
            assert original is not None, "Original required for side_by_side mode"
            img = self.side_by_side(depth, original)
            Image.fromarray(img).save(out_path)
        else:
            img = self.colourise(depth)
            Image.fromarray(img).save(out_path)

        log.info("Saved depth visualisation -> %s", out_path)
        return out_path
    
# ----- Internal ------------------------------------------------------------------------

@staticmethod
def _validate(depth: np.ndarray) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth map, got shape {depth.shape}")
    return depth.astype(np.float32)

