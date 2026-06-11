""" Central configuration and constants for depth estimation."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class ModelBackend(str, Enum):
    MIDAS = "midas"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"

class MiDaSModel(str, Enum):
    SMALL = "MiDaS_small"
    DPT_HYBRID = "DPT_Hybrid"
    DPT_LARGE = "DPT_Large"

class DepthAnythingSize(str, Enum):
    SMALL = "vits"
    BASE = "vitb"
    LARGE = "vitl"

class ColormapName(str, Enum):
    INFRNO = "inferno"
    MAGMA = "magma"
    PLASMA = "plasma"
    VIRIDIS = "viridis"
    TURBO = "turbo"
    GRAY = "gray"

class Settings(BaseSettings):

    # ----- Model section --------------------------------------------------
    backend: ModelBackend = ModelBackend.DEPTH_ANYTHING_V2
    midas_model: MiDaSModel = MiDaSModel.DPT_HYBRID
    depth_anything_size: DepthAnythingSize = DepthAnythingSize.SMALL

    # ----- Inference ------------------------------------------------------
    input_size: int = Field(default=518, ge=64, le=1024)
    batch_size: int = Field(default=1, ge=1)

    # ----- Visualisation --------------------------------------------------
    colormap: ColormapName = ColormapName.INFRNO
    invert_depth: bool = True  #far=dark, near=bright when True
    alpha_blend: float = Field(default=0.6, ge=0.0, le=1.0)

    # ----- Webcam / live demo ---------------------------------------------
    webcam_device_id: int = 0
    webcam_width: int = 640
    webcam_height: int = 480
    webcam_fps: int = 30

    # ----- Gradio app -----------------------------------------------------
    app_host: str = "127.0.0.1"
    app_port: int = 7860
    app_share: bool = False
    app_theme: Literal["dark", "light", "default"] = "default"

    # ----- Logging --------------------------------------------------------
    log_level: str = "INRO"
    log_file: str | None = None

    class Config:
        env_prefix = "DEPTH_" # ALL environment variables must start with DEPTH_ other wise not override the settngs fields
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# UI names 
BACKEND_LABELS: dict[ModelBackend, str] = {
    ModelBackend.MIDAS: "MiDaS",
    ModelBackend.DEPTH_ANYTHING_V2: "Depth Anything v2"
}

COLORMAP_LABELS: dict[ColormapName, str] = {
    ColormapName.INFRNO: "Inferno 🔥",
    ColormapName.MAGMA: "Magma 🌋",
    ColormapName.PLASMA: "Plasma ☄️",
    ColormapName.VIRIDIS : "Viridis ☘️",
    ColormapName.TURBO: "Turbo 🌈",
    ColormapName.GRAY: "Grayscale 🔘",
}