"""Model sub-package - estimators and factory helper."""

from __future__ import annotations

import torch

from depth_estimation.config import settings,ModelBackend
from depth_estimation.model.base_estimator import BaseDepthEstimator, ImageLike
from depth_estimation.model.depth_anything_estimator import DepthAnythingV2Estimator
from depth_estimation.model.midas_estimator import MiDaSEstimator

def build_estimator(backend: ModelBackend | str | None = None, device: torch.device | None = None) -> BaseDepthEstimator:
    """Returns the right estimator based on the backend."""

    effective = ModelBackend(backend) if backend else settings.backend

    if effective == ModelBackend.MIDAS:
        return MiDaSEstimator(
            model_type=settings.midas_model,
            device=device
        )
    if effective == ModelBackend.DEPTH_ANYTHING_V2:
        return DepthAnythingV2Estimator(
            size=settings.depth_anything_size,
            device=device
        )
    raise ValueError(f"Unknown backend: {effective!r}")

__all__ = [
    "MiDaSEstimator",
    "DepthAnythingV2Estimator",
    "BaseDepthEstimator",
    "ImageLike",
    "build_estimator",
]
