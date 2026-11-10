"""Proof run for the Backbone Builder trainer on generated images (Bitcrush ISSUE-1655).

Draws a small labelled dataset, trains a backbone on it through the real entry
point (``run_backbone_training``: sampling plan, augmentation, AMP, EMA,
validation, candidate export) and reports whether the run held together and how
fast it went. No real image is read, so the same command proves the trainer at
home and on a rented GPU::

    python -m bittrainer.tools.backbone_proof
    python -m bittrainer.tools.backbone_proof --size base --image-size 224 \
        --images 4000 --epochs 2 --batch-size 64 --workers 4 --init timm

The images carry signals a ConvNeXt learns quickly and that survive the
trainer's augmentation: a three-way ``layout`` group (two halves side by side,
two halves stacked, or a blob in the centre) and two binary concepts (diagonal
``striped`` lines, a ``ringed`` outline). Some labels are withheld on purpose, so
the masked loss is exercised the way a partly-labelled pool exercises it.

Exit code 0 when every check passes, 1 otherwise. The printed report holds
numbers and settings only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import tempfile
import time
from pathlib import Path

CONCEPTS = ("striped", "ringed")
GROUP = "layout"
LAYOUTS = ("side_by_side", "stacked", "centre_blob")

# Macro-F1 the WEAKER head family (concepts or groups) must reach on a default
# run. Chance is about 0.5 for the concepts; two default CPU runs from random
# weights landed at 0.74 and 0.78 for the concepts and 1.0 for the group.
DEFAULT_MIN_SCORE = 0.6
DEFAULT_EPOCHS = 10

_AMP_CHOICES = {"auto": "auto", "fp16": "float16", "bf16": "bfloat16"}


def _tone(rng: random.Random, low: int, high: int) -> tuple[int, int, int]:
    return (rng.randint(low, high), rng.randint(low, high), rng.randint(low, high))


def _draw(
    index: int, image_size: int, seed: int, layout: str, striped: bool, ringed: bool
):
    """One fixture image. Tones stay mid-range so RandAugment's solarize and
    posterize leave the signal standing; which half is the dark one is random,
    so only the arrangement carries the ``layout`` label."""
    from PIL import Image, ImageDraw

    rng = random.Random(f"{seed}|{index}")
    side = max(32, round(image_size * 1.15))
    aspect = rng.choice((1.0, 1.25, 0.8))
    width, height = (
        (round(side * aspect), side) if aspect >= 1 else (side, round(side / aspect))
    )

    first, second = _tone(rng, 30, 80), _tone(rng, 130, 175)
    if rng.random() < 0.5:
        first, second = second, first
    image = Image.new("RGB", (width, height), first)
    draw = ImageDraw.Draw(image)
    if layout == "side_by_side":
        draw.rectangle([width // 2, 0, width, height], fill=second)
    elif layout == "stacked":
        draw.rectangle([0, height // 2, width, height], fill=second)
    else:
        rx, ry = round(width * 0.3), round(height * 0.3)
        cx, cy = width // 2, height // 2
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=second)

    # Both marks scale with the image and stay several pixels wide, so they
    # survive the crop, the resize and JPEG at any training size.
    if striped:
        gap = max(6, side // 5)
        ink = _tone(rng, 85, 125)
        for offset in range(-height, width, gap):
            draw.line(
                [(offset, 0), (offset + height, height)],
                fill=ink,
                width=max(2, side // 14),
            )
    if ringed:
        radius = round(min(width, height) * 0.42)
        cx, cy = width // 2, height // 2
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=_tone(rng, 95, 115),
            width=max(3, side // 10),
        )
    return image


def make_fixture_dataset(
    root: Path, *, count: int, image_size: int, seed: int = 0
) -> list[dict]:
    """Write ``count`` generated images under ``root`` and return their records
    in the ``run_backbone_training`` request shape. Same arguments, same bytes
    and labels. Labels cycle through every combination; roughly one in five of
    each head's labels is withheld (never all of an image's labels)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index in range(count):
        layout = LAYOUTS[index % 3]
        striped = (index // 3) % 2 == 0
        ringed = (index // 6) % 2 == 0
        path = root / f"proof_{index:06d}.jpg"
        _draw(index, image_size, seed, layout, striped, ringed).save(path, quality=90)

        binary: dict[str, str] = {}
        if index % 5 != 1:
            binary["striped"] = "positive" if striped else "negative"
        if index % 5 != 3:
            binary["ringed"] = "positive" if ringed else "negative"
        groups = {} if index % 7 == 2 else {GROUP: layout}
        records.append(
            {
                "content_hash": hashlib.sha256(
                    f"backbone-proof|{seed}|{index}".encode()
                ).hexdigest(),
                "file_paths": [str(path)],
                "binary": binary,
                "groups": groups,
                "splits": {"train": 1},
            }
        )
    return records


def build_request(
    root: Path,
    records: list[dict],
    *,
    size: str = "atto",
    image_size: int = 64,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 32,
    device: str | None = None,
    amp: str = "auto",
    workers: int = 0,
    channels_last: bool = False,
    init: str = "random",
    stock_grn: bool = False,
) -> dict:
    """The ``run_backbone_training`` request for a proof run. Every training
    default the Backbone Builder ships with is left alone; only the settings
    named here are set."""
    config: dict = {
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "validation_split": 0.2,
        "dataloader_workers": int(workers),
        "channels_last": bool(channels_last),
    }
    if stock_grn:  # timm's own GRN layer instead of the trainer's lean one
        config["lean_grn"] = False
    if device:
        config["device"] = device
    if amp == "off":
        config["use_amp"] = False
    else:
        config["use_amp"] = True
        config["amp_dtype"] = _AMP_CHOICES[amp]
    backbone_init = (
        {"source": "timm_pretrained"}
        if init == "timm"
        else {"source": "random_init", "checkpoint_path": None}
    )
    return {
        "run_id": "backbone_proof",
        "family_name": "bitcrush_backbone",
        "architecture": "convnextv2",
        "size_alias": size,
        "display_size": size,
        "convnextv2_size": size,
        "candidate_checkpoint_path": str(
            Path(root) / "candidate" / "backbone_proof.safetensors"
        ),
        "records": records,
        "training_config": config,
        "backbone_init": backbone_init,
        "license_provenance": "proof_run",
        "external_pretrained_used": init == "timm",
    }


def _checkpoint_finite(path: str) -> bool:
    import torch
    from safetensors.torch import load_file

    state = load_file(path)
    return bool(state) and all(
        bool(torch.isfinite(tensor).all())
        for tensor in state.values()
        if tensor.is_floating_point()
    )


def run_proof(
    root: Path,
    *,
    images: int = 600,
    image_size: int = 64,
    size: str = "atto",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 32,
    device: str | None = None,
    amp: str = "auto",
    workers: int = 0,
    channels_last: bool = False,
    init: str = "random",
    stock_grn: bool = False,
    min_score: float | None = DEFAULT_MIN_SCORE,
    seed: int = 0,
) -> dict:
    """Generate the fixtures under ``root``, train on them and return the report.

    A failing run is reported, not raised: ``checks`` says which part did not
    hold and ``error`` carries the exception text. ``min_score=None`` skips the
    learning floor (for runs too short to learn anything).
    """
    import torch

    from bittrainer.backbone_trainer import run_backbone_training

    root = Path(root)
    started = time.perf_counter()
    records = make_fixture_dataset(
        root / "images", count=images, image_size=image_size, seed=seed
    )
    request = build_request(
        root,
        records,
        size=size,
        image_size=image_size,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        amp=amp,
        workers=workers,
        channels_last=channels_last,
        init=init,
        stock_grn=stock_grn,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    frames: list[dict] = []
    result: dict | None = None
    error: str | None = None
    try:
        result = asyncio.run(
            run_backbone_training(request, progress_callback=frames.append)
        )
    except Exception as exc:  # noqa: BLE001 - the report is the product of a proof run
        error = f"{type(exc).__name__}: {exc}"

    preparing = next(
        (f for f in frames if f.get("stage") == "preparing" and "train_samples" in f),
        {},
    )
    epoch_rows = [
        {
            key: frame.get(key)
            for key in (
                "epoch",
                "loss",
                "validation_score",
                "train_seconds",
                "train_images",
                "images_per_second",
                "skipped_steps",
                "grad_scale",
            )
        }
        for frame in frames
        if frame.get("stage") == "training" and "loss" in frame
    ]
    losses = [row["loss"] for row in epoch_rows]
    speeds = [
        row["images_per_second"] for row in epoch_rows if row["images_per_second"]
    ]
    completed = result is not None and not result.get("paused")
    score = float(result["validation_score"]) if completed else None
    metrics = (result or {}).get("validation_metrics") or {}
    # The learning floor is held against the WEAKER head family. The selection
    # score averages the two, so a perfect group head could carry concept heads
    # that learned nothing straight past it.
    families = [
        float(metrics[key])
        for key in ("binary_macro_f1", "group_macro_f1")
        if key in metrics
    ]
    weakest = min(families) if families else score

    device_type = preparing.get("device") or (device or "unknown")
    gpu_name = capability = peak_vram_mb = None
    if device_type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device or "cuda"))
        gpu_name = props.name
        capability = f"{props.major}.{props.minor}"
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 2**20)

    checks = {
        "completed": completed,
        "loss_finite": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "loss_fell": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
        "checkpoint_finite": _checkpoint_finite(result["candidate_checkpoint_path"])
        if completed
        else False,
        "score_reached": None
        if min_score is None
        else bool(completed and weakest is not None and weakest >= min_score),
    }
    return {
        "torch": torch.__version__,
        "device": device_type,
        "gpu_name": gpu_name,
        "compute_capability": capability,
        "size": size,
        "init": init,
        "image_size": image_size,
        "batch_size": batch_size,
        "images": images,
        "train_images": preparing.get("train_samples"),
        "val_images": preparing.get("val_samples"),
        "workers": workers,
        "channels_last": channels_last,
        "amp_dtype": preparing.get("amp_dtype"),
        "grad_scaler": preparing.get("grad_scaler"),
        "lean_grn": preparing.get("lean_grn"),
        "epochs": epoch_rows,
        "best_images_per_second": max(speeds) if speeds else None,
        "peak_vram_mb": peak_vram_mb,
        # fp16 only: optimizer steps the GradScaler skipped over the whole run.
        "skipped_steps": (result or {}).get("skipped_steps"),
        "validation_score": score,
        "weakest_family_f1": weakest,
        "min_score": min_score,
        "validation_metrics": {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        "seconds": round(time.perf_counter() - started, 1),
        "error": error,
        "checks": checks,
        "passed": all(value for value in checks.values() if value is not None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bittrainer.tools.backbone_proof",
        description="Train a backbone on generated images and report whether it held together.",
    )
    parser.add_argument(
        "--size", default="atto", help="ConvNeXt V2 size (atto ... large)"
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--images", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument(
        "--amp", default="auto", choices=["auto", "fp16", "bf16", "off"]
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--stock-grn",
        action="store_true",
        help="train with timm's own GRN layer (slower in half precision)",
    )
    parser.add_argument("--init", default="random", choices=["random", "timm"])
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="validation score the run must reach; a negative value skips the check",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workdir",
        default=None,
        help="keep the images and candidate here (default: temp, deleted)",
    )
    args = parser.parse_args(argv)

    options = {
        "size": args.size,
        "image_size": args.image_size,
        "images": args.images,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "amp": args.amp,
        "channels_last": args.channels_last,
        "stock_grn": args.stock_grn,
        "init": args.init,
        "min_score": None if args.min_score < 0 else args.min_score,
        "device": None if args.device == "auto" else args.device,
        "seed": args.seed,
    }
    if args.workdir:
        report = run_proof(Path(args.workdir), **options)
    else:
        with tempfile.TemporaryDirectory(prefix="backbone_proof_") as workdir:
            report = run_proof(Path(workdir), **options)

    print(json.dumps(report, indent=2, default=str))
    for name, value in (report.get("checks") or {}).items():
        state = "skipped" if value is None else ("ok" if value else "FAILED")
        print(f"  {name}: {state}")
    passed = bool(report.get("passed"))
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
