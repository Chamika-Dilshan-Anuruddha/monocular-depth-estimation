""" CLI script - run depth estimation over a folder of CSV of image paths."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import numpy as np
from PIL import Image

from depth_estimation.config import ColormapName, MiDaSModel, ModelBackend
from depth_estimation.model import build_estimator
from depth_estimation.pipeline import DepthPostprocessor, ImagePreProcessor
from depth_estimation.utils import get_logger

log = get_logger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def collect_images(input_dir: Path | None, csv_file: Path | None) -> list[Path]:
    paths: list[Path] = []
    if input_dir:
        for p in sorted(input_dir.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXT:
                paths.append(p)
    if csv_file:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = Path(row.get("path", "").strip())
                if p.exists():
                    paths.append(p)
                else:
                    log.warning("CSV path not found: %s",p)
    if not paths:
        log.error("No images found. Check --input / --csv.")
        sys.exit(1)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch monocular depth estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, metavar="DIR", help="Directory of image")
    src.add_argument("--csv", type=Path, metavar="FILE", help="CSV file with a 'path' column")

    parser.add_argument("--output", "-o", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--backend", choices=[b.value for b in ModelBackend])
    parser.add_argument("--model", default=None, help="MiDaS variant or DA-v2 size (vits/vitb/vitl or MiDaS_small/DPT_Hybrid/DPT_Large)")
    parser.add_argument("--colormap", choices=[c.value for c in ColormapName], default=ColormapName.INFERNO.value)
    parser.add_argument("--mode", choices=["colour", "side_by_side", "blend"], default="colour")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--no-invert", action="store_true", help="Do Not invert depth (far = dark)")
    parser.add_argument("--max-dim", type=int, default=1024, help="Resize longest edge to this before inference")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # ----- Build model -------------------------------------------
    if args.model is None:
        variant = None
    else:
        variant = args.model 
    
    from depth_estimation.config import settings, DepthAnythingSize
    if variant:
        if args.backend == ModelBackend.MIDAS.value:
            settings.midas_model = MiDaSModel(variant)
        else:
            settings.depth_anything_size = DepthAnythingSize(variant)
    
    estimator = build_estimator(backend=args.backend)
    estimator.ensure_loaded()
    log.info("Using estimator: %s", estimator)

    pre = ImagePreProcessor(max_dimension=args.max_dim)
    post = DepthPostprocessor(
        colormap=ColormapName(args.colormap),
        invert=not args.no_invert,
        alpha=args.alpha,
    )

    # ----- Collect images ----------------------------------------
    image_paths = collect_images(args.input, args.csv)
    log.info("Processing %d image(s) ...", len(image_paths))

    total_ms = 0.0
    for idx, img_path in enumerate(image_paths,1):
        t0 = time.perf_counter()
        try:
            rgb = pre.load(img_path)
            depth = estimator.predict(rgb)

            out_name = img_path.stem + "_depth.png"
            out_path = args.output / out_name

            post.save(depth, out_path, original=rgb, mode=args.mode)
            elapsed = (time.perf_counter() - t0) * 1000
            total_ms += elapsed
            log.info("[%d/%d] %s -> %s (%.0f ms)", idx, len(image_paths), img_path.name, out_name, elapsed)

        except Exception as e:
            log.error("[%d/%d] Falled on %s: %s", idx, len(image_paths), img_path, e)

    avg = total_ms / len(image_paths)
    log.info("Done. Average: %.0f ms/image. Results in: %s", avg, args.output)


if __name__ == "__main__":
    main()
