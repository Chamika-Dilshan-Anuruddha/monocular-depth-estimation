"""MiDaS monocular depth estimator (Inter ISL)."""

from __future__ import annotations

import numpy as np
import torch

from depth_estimation.config import settings, MiDaSModel
from depth_estimation.model.base_estimator import BaseDepthEstimator
from depth_estimation.utils import get_logger

log = get_logger(__name__)

_HUB_REPO = "intel-isl/MiDas"

class MiDaSEstimator(BaseDepthEstimator):

    def __init__(self, model_type: MiDaSModel | str = MiDaSModel.DPT_HYBRID, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self.model_type = (model_type.value if isinstance(model_type, MiDaSModel) else model_type)
        self._model: torch.nn.Module | None = None
        self._transform = None

    # ----- Lifecycle ---------------------------------------------------------------------------------------------

    def load(self) -> None:
        log.info("Downloading / loadng MiDaS '%s' from torch.hub ...", self.model_type)
        self._model = torch.hub.load(
            _HUB_REPO,
            self.model_type,
            pretrained=True,
            trust_repo=True
        )
        self._model.to(self.device).eval()
        # Preprocessing functions for the mdoel
        midas_transforms = torch.hub.load(
            _HUB_REPO,
            "transforms",
            trust_repo=True
        )
        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform
        
        log.info("MiDaS '%s' loaded successfully.", self.model_type)

    # ----- Inference -----------------------------------------------------------------------------------------------

    def _infer(self, image: np.ndarray) -> np.ndarray:

        assert self._model is not None and self._transform is not None
        
        input_batch = self._transform(image).to(self.device)

        with torch.inference_mode():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        d_min, d_max = depth.min(), depth.max()
        # normalize the depth to 0 to 1
        if d_max - d_min > 1e-6:
            depth = (depth -d_min)/(d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth
    
    # ----- Properties --------------------------------------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"MiDaS/{self.model_type}"
    
    def __repr__(self) -> str:
        status = "loaded" if self._model_loaded else "not loaded"
        return f"MiDaSEstimator(model= {self.model_type}, device={self.device}, {status})"