""" Geadio web application - monocular depth estimation."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

import numpy as np
import gradio as gr
from PIL import Image

from app.components.ui_components import (
    alpha_slider,
    colormap_selector,
    device_inofo_block,
    display_model_selector,
    invert_toggle,
    model_selector_block
)

from depth_estimation.config import (
    ColormapName,
    DepthAnythingSize,
    MiDaSModel,
    ModelBackend,
    settings
)

from depth_estimation.model import BaseDepthEstimator, build_estimator
from depth_estimation.pipeline import DepthPostprocessor, ImagePreProcessor
from depth_estimation.utils import device_info, get_logger

log = get_logger(__name__)

_estimator_cache: dict[str, BaseDepthEstimator] = {}

def _get_estimator(backend: str, variant: str) -> BaseDepthEstimator:
    key = f"{backend}:{variant}"
    if key not in _estimator_cache:
        log.info("Building estimator %s ...", key)
        if backend == ModelBackend.MIDAS.value:
            from depth_estimation.model import MiDaSEstimator
            est = MiDaSEstimator(model_type=variant)
        else:
            from depth_estimation.model import DepthAnythingV2Estimator
            est = DepthAnythingV2Estimator(size=variant)
        est.ensure_loaded()
        _estimator_cache[key] = est
    return _estimator_cache[key]

# ---------- Core inference helper -----------------------------------------------------------------------------------------------

def run_depth(
        image: np.ndarray,
        backend: str,
        variant: str,
        colormap: str,
        display_mode: str,
        alpha: float,
        invert: bool
) -> tuple[np.ndarray, str]:
    """Run bepth estimation and return (visualisation_array, stats_text)"""

    if image is None:
        return np.zeros((256,256,3), dtype=np.uint8), "No image provided"
    
    t0 = time.perf_counter()
    estimator = _get_estimator(backend,variant)
    pre = ImagePreProcessor(max_dimension=1024)
    rgb = pre.load(image)

    depth = estimator.predict(rgb)

    post = DepthPostprocessor(
        colormap=ColormapName(colormap),
        invert=invert,
        alpha=alpha
    )

    if display_mode == "side_by_side":
        out = post.side_by_side(depth, rgb)
    elif display_mode == "blend":
        out = post.blend(depth,rgb)
    else:
        out = post.colourise(depth)

    elapsed = (time.perf_counter() - t0) * 1000
    stats = post.depth_stats(depth)
    info = (
        f" {elapsed: .0f} ms | "
        f"min={stats['min']:.3f} max={stats['max']:.3f} "
        f"mean={stats['mean']:.3f} std={stats['std']}"
    )
    return out,info

# ----------- Webcam streaming callback -------------------------------------------------------------------------------

def run_depth_webcam(
        frame: np.ndarray,
        backend: str,
        variant: str,
        colormap: str,
        display_mode: str,
        alpha: float,
        invert: bool,
) -> np.ndarray:
    
    if frame is None:
        return None
    
    out, _ = run_depth(frame,backend, variant, colormap, display_mode, alpha, invert)
    return out

# ----- Visibility toggle helper -----------------------------------------------------------------------------------------

def _toggle_variant_visibility(backend: str):
    is_da = backend == ModelBackend.DEPTH_ANYTHING_V2.value
    return gr.update(visible=is_da), gr.update(visible=not is_da)

def _resolver_variant(backend: str, da_size: str, midas_variant: str) -> str:
    if backend == ModelBackend.DEPTH_ANYTHING_V2.value:
        return da_size
    return midas_variant


# ----- Build UI ----------------------------------------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    div_info = device_info()

    with gr.Blocks(
        title="Monocular Depth Estimation",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
        css="""
        .title-area { text-align: center; padding: 1.5rem 0 0.5rem; }
        .stats-box { font-family: monospace; font-size: 0.85rem; }
        """
    ) as demo:
        
        # Header
        gr.HTML(
            """
                <div class="title-area">
                 <h1 style="font-size:2rem; font-weight:700; margin:0">
                     Monocular Depth Estimation
                 </h1>
                 <p style="color:#888; margin-top:0.25rem">
                    MiDaS &amp; Depth Anything v2 - single-image and live webcam demo
                 </p>
            </div>
            """
        )

        # Shared controls (dicebar)
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Model")
                backend, da_size, midas_variant = model_selector_block()

                gr.Markdown("### Visualisation")
                colormap = colormap_selector()
                disp_mode = display_model_selector()
                alpha = alpha_slider()
                invert = invert_toggle()

                with gr.Accordion("System info", open=False):
                    gr.Markdown(
                        "\n\n".join(f"**{k}**: `{v}`" for k,v in div_info.items())
                    )
                
            # Tabs
            with gr.Column(scale=3):
                with gr.Tabs():

                    # Tab 1: single image
                    with gr.TabItem("Single Image"):
                        with gr.Row():
                            img_in = gr.Image(
                                label="Input Image",
                                type="numpy",
                                height=380,
                            )

                            img_out = gr.Image(
                                label="Depth visalisation",
                                type="numpy",
                                height=380,
                            )

                        with gr.Row():

                            run_btn = gr.Button("Estimate Depth", variant="primary")
                            stats_md = gr.Markdown(
                                value="Run inference to see stats.",
                                elem_classes=["stats-box"]
                            )

                            example_data = [
                                ["examples/outdoor.jpg"],
                                ["examples/indoor.jpg"],
                                ["examples/portrait.jpg"],
                            ]

                            try:
                                gr.Examples(
                                    examples=example_data,
                                    inputs=[img_in],
                                    label="Example images"
                                )
                            except Exception:
                                pass

                            def _infer_image(img, be, da, mi, cm, dm, al, iv):
                                variant = _resolver_variant(be, da ,mi)
                                return run_depth(img, be, variant, cm, dm, al, iv)
                            
                            run_btn.click(
                                fn=_infer_image,
                                inputs=[
                                    img_in, backend, da_size, midas_variant,
                                    colormap, disp_mode, alpha, invert ],
                                    outputs=[img_out, stats_md]
                            )
                    
                    # Tab 2: live webcam
                    with gr.TabItem("Live Webcam"):
                        gr.Markdown(
                            "> Enable your webcam below. ",
                            "Frame are processed in real-time"
                        )

                        webcam_in = gr.Image(
                            label="Webcam feed",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            height=380
                        )

                        webcam_out = gr.Image(
                            label="Live depth",
                            type="numpy",
                            height=380,
                            streaming=True
                        )

                        def _infer_webcam(frame, be, da, mi, cm, dm, al, iv):
                            variant = _resolver_variant(be, da, mi)
                            return run_depth_webcam(
                                frame, be, variant, cm, dm, al, iv
                            )
                        
                        webcam_in.stream(
                            fn=_infer_webcam,
                            inputs=[webcam_in, backend, da_size, midas_variant,
                                    colormap, disp_mode, alpha, invert],
                            outputs=[webcam_out]
                        )
        
        backend.change(
            fn=_toggle_variant_visibility,
            inputs=[backend],
            outputs=[da_size, midas_variant]
        )

    return demo


# ----- Entry point ------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        share=settings.app_share,
    )