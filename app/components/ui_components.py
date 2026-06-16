""" Reusable Gradio blocks for the depth estimation."""

from __future__ import annotations

import gradio as gr

from depth_estimation.config import (
    BACKEND_LABELS,
    COLORMAP_LABELS,
    ColormapName,
    DepthAnythingSize,
    MiDaSModel,
    ModelBackend
)

def model_selector_block() -> tuple[gr.Radio, gr.Radio, gr.Radio]:
    backend = gr.Radio(
        choices=[(v,k.value) for k,v in BACKEND_LABELS.items()],
        value=ModelBackend.DEPTH_ANYTHING_V2.value,
        label="Backend",
        info="Depth Anything v2 is recommended for most use-cases"
    )

    da_size = gr.Radio(
        choices=[
            ("Small (fastest, -25 M)", DepthAnythingSize.SMALL.value),
            ("Base (balanced, -97 M)", DepthAnythingSize.BASE.value),
            ("Large (best quality, -335 M)", DepthAnythingSize.LARGE.value),

        ],
        value=DepthAnythingSize.SMALL.value,
        label="Depth Anything v2-size",
        visible=True,

    )

    midas_variant = gr.Radio(
        choices=[
            ("Small (real-time)", MiDaSModel.SMALL.value),
            ("DPT-Hybrid (recommended)", MiDaSModel.DPT_HYBRID.value),
            ("DPT-Large (best quality)", MiDaSModel.DPT_LARGE.value),

        ],
        value=MiDaSModel.DPT_HYBRID.value,
        label="MiDas variant",
        visible=False

    )

    return backend, da_size, midas_variant

def colormap_selector() -> gr.Dropdown:
    return gr.Dropdown(
        choices=[(v, k.value) for k,v in COLORMAP_LABELS.items()],
        value=ColormapName.INFERNO.value,
        label="Colormap"
    ) 

def display_model_selector() -> gr.Radio:
    return gr.Radio(
        choices=[
            ("Depth only", "colour"),
            ("Side-by-side", "side_by_side"),
            ("Blended overlay", "blend"),
        ],
        value="side_by_side",
        label="Display mode"
    )

def alpha_slider() -> gr.Slider:
    return gr.Slider(
        minimum=0,
        maximum=1.0,
        value=0.6,
        step=0.05,
        label="Blend a (depth weight)",
        info="Only applies in Blended overlay mode,"
    )

def invert_toggle() -> gr.Checkbox:
    return gr.Checkbox(
        value=True,
        label="Invert depth (near = dark)."
    )


def device_inofo_block(info: dict) -> gr.Markdown:
    lines = [f"**{k}**: `{v}`" for k,v in info.items()]
    return gr.Markdown("\n\n".join(lines))