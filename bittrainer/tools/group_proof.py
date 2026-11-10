"""Proof run for the group trainers on generated images (Bitcrush ISSUE-1690).

Draws a small group, then runs what a cloud group training runs, end to end:

1. a head-only training on the drawn images: the model Engine holds at home;
2. a scoped export of the group's cached tensors
   (``bittrainer.cloud_export.build_scoped_export``), verified the way the pod
   verifies what it receives;
3. the drawn images deleted, and a full update of the head-only model trained
   from the export alone (``sourceless``, as on the pod).

It reports how fast the update trained and how much memory it took, the
precision it chose, whether the GradScaler and the lean GRN layer were on,
whether the fused back pass ran, and the optimizer steps the scaler skipped.
No real image is read, so the same command proves the trainers at home and on
the rented T4::

    python -m bittrainer.tools.group_proof
    python -m bittrainer.tools.group_proof --size base --bucket-size 512 --device cuda

The classes are arrangements a ConvNeXt learns quickly (two halves side by
side, two halves stacked, a blob in the centre) and ``__none__`` (a plain
frame with a few faint lines). Images come in three aspect ratios, so the run
trains three buckets.

``--pause-after-steps N`` pauses the update at its N-th optimizer step and
resumes it from its backup. ``--expect-refusal`` turns the check round: the
update must be refused before it trains (an oversize ``--batch-size``) and
leave no backup.

Exit code 0 when every check passes, 1 otherwise. The printed report holds
numbers and settings only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import threading
import time
from pathlib import Path

CLASSES = ("side_by_side", "stacked", "centre_blob", "__none__")
GROUP = "proof_group"

# What ``--amp`` asks the trainer for. "auto" is the trainer's own default,
# bf16, which a card without native bf16 (the T4) turns into fp16.
_AMP_DTYPES = {"auto": "bfloat16", "fp16": "float16", "off": "float32"}


# --------------------------------------------------------------------------- #
# The group                                                                   #
# --------------------------------------------------------------------------- #


def _draw(name: str, index: int, image_size: int, seed: int):
    from bittrainer.tools.backbone_proof import _draw as draw_layout
    from bittrainer.tools.backbone_proof import _tone

    if name != "__none__":
        return draw_layout(index, image_size, seed, name, striped=False, ringed=False)
    from PIL import Image, ImageDraw

    rng = random.Random(f"{seed}|none|{index}")
    side = max(32, round(image_size * 1.15))
    aspect = rng.choice((1.0, 1.25, 0.8))
    width, height = (
        (round(side * aspect), side) if aspect >= 1 else (side, round(side / aspect))
    )
    image = Image.new("RGB", (width, height), _tone(rng, 30, 175))
    draw = ImageDraw.Draw(image)
    for _ in range(3):
        draw.line(
            [(rng.randrange(width), rng.randrange(height)),
             (rng.randrange(width), rng.randrange(height))],
            fill=_tone(rng, 60, 150),
            width=max(1, side // 40),
        )  # fmt: skip
    return image


def make_group_folder(
    root: Path, *, per_class: int, val_per_class: int, image_size: int, seed: int = 0
) -> dict[str, int]:
    """Draw ``root/<class>/<split>/*.jpg`` for every class in :data:`CLASSES`.
    Same arguments, same bytes. Returns how many images each split holds."""
    root = Path(root)
    counts = {"train": 0, "val": 0}
    for number, name in enumerate(CLASSES):
        for split, count in (("train", per_class), ("val", val_per_class)):
            folder = root / name / split
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                drawn = _draw(name, number * 100_000 + index + (50_000 if split == "val" else 0),
                              image_size, seed)  # fmt: skip
                drawn.save(folder / f"{name}_{index:05d}.jpg", quality=90)
                counts[split] += 1
    return counts


def _backbone_init(init: str, trunk: str | None) -> dict | None:
    if trunk:
        return {"source": "local_candidate", "checkpoint_path": str(trunk)}
    if init == "timm":
        return None  # timm's pretrained weights, the trainer's default
    return {"source": "random_init", "checkpoint_path": None}


def build_config(
    group_folder: Path,
    checkpoint_dir: Path,
    *,
    cache_dir: Path,
    embedding_dir: Path,
    size: str = "atto",
    bucket_size: int = 512,
    epochs: int = 2,
    batch_size: int = 0,
    amp: str = "auto",
    device: str = "cpu",
    compile: bool = True,
    stock_grn: bool = False,
    init: str = "random",
    trunk: str | None = None,
    workers: int = 4,
    from_scratch: bool = False,
    sourceless: bool = False,
    preflight: bool | None = None,
    backup_dir: Path | None = None,
    resume_from: Path | None = None,
):
    """The ``GroupTrainConfig`` for one phase of a proof run. Every training
    default the group trainer ships with is left alone; only the settings
    named here are set (``batch_size`` 0 is the automatic batch size)."""
    from bittrainer.group_trainer import GroupTrainConfig

    return GroupTrainConfig(
        group_folder=str(group_folder),
        num_classes=len(CLASSES),
        class_names=list(CLASSES),
        checkpoint_dir=str(checkpoint_dir),
        group_name=GROUP,
        max_epochs=int(epochs),
        patience=max(3, int(epochs)),
        backbone_variant=size,
        train_resolution=int(bucket_size),
        device=device,
        dtype=_AMP_DTYPES[amp],
        batch_size=int(batch_size) or None,
        use_cache=True,
        cache_dir=str(cache_dir),
        cache_workers=max(1, int(workers)),
        embedding_cache_dir=str(embedding_dir),
        use_compile=bool(compile),
        lean_grn=not stock_grn,
        preflight=preflight,
        dataloader_workers=int(workers),
        backbone_init=_backbone_init(init, trunk),
        from_scratch=from_scratch,
        sourceless=sourceless,
        backup_dir=str(backup_dir) if backup_dir else None,
        resume_from=str(resume_from) if resume_from else None,
    )


# --------------------------------------------------------------------------- #
# Reading what a run said                                                     #
# --------------------------------------------------------------------------- #


def _texts(frames: list[dict]) -> list[str]:
    return [str(f.get("status_text") or "") for f in frames]


def update_numbers(frames: list[dict]) -> dict:
    """What the full update reported: its epochs, its speed, its batch size,
    what its preflight measured, its back-pass mode and whether it compiled."""
    epochs = [
        {
            key: frame.get(key)
            for key in (
                "epoch",
                "train_loss",
                "val_macro_f1",
                "elapsed_seconds",
                "skipped_steps",
                "grad_scale",
            )
        }
        for frame in frames
        if frame.get("type") == "epoch_complete"
    ]
    # The last step frame of an epoch covers the whole epoch's training.
    speed: dict = {}
    for frame in frames:
        if frame.get("stage") == "training" and frame.get("images_per_s"):
            speed[frame.get("epoch")] = frame["images_per_s"]
    for row in epochs:
        row["images_per_second"] = speed.get(row["epoch"])
    texts = _texts(frames)
    numbers: dict = {
        "epochs": epochs,
        "best_images_per_second": max(speed.values()) if speed else None,
        "batch_size_used": next(
            (f.get("batch_size") for f in frames if f.get("type") == "autobatch"), None
        ),
        "preflight_mb": next(
            (f.get("peak_vram_mb") for f in frames if f.get("stage") == "preflight"),
            None,
        ),
        "fused_back_pass": None,
        "compiled": any(t.startswith("Compiling") for t in texts)
        and not any("Compilation failed" in t or "torch.compile unavailable" in t for t in texts),
        "resumed": any(f.get("type") == "training_resumed" for f in frames),
    }
    for text in texts:
        if text.startswith("Fused back pass"):
            numbers["fused_back_pass"] = text
    return numbers


def _checkpoint_finite(path: str | None) -> bool:
    import torch

    if not path or not Path(path).is_file():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    state = data.get("state_dict", data) if isinstance(data, dict) else {}
    return bool(state) and all(
        bool(torch.isfinite(t).all())
        for t in state.values()
        if isinstance(t, torch.Tensor) and t.is_floating_point()
    )


def _no_backup(folder: Path) -> bool:
    return not list(Path(folder).glob("backup_*.pt*"))


# --------------------------------------------------------------------------- #
# The run                                                                     #
# --------------------------------------------------------------------------- #


def run_proof(
    root: Path,
    *,
    size: str = "atto",
    bucket_size: int = 512,
    images_per_class: int = 40,
    epochs: int = 2,
    batch_size: int = 0,
    device: str | None = None,
    amp: str = "auto",
    compile: bool = True,
    stock_grn: bool = False,
    init: str = "random",
    trunk: str | None = None,
    workers: int = 4,
    pause_after_steps: int = 0,
    preflight: bool | None = None,
    expect_refusal: bool = False,
    min_score: float | None = None,
    seed: int = 0,
) -> dict:
    """Draw the group under ``root``, run the three phases and return the
    report. A failing run is reported, not raised: ``checks`` says which part
    did not hold and ``error`` carries the exception text."""
    import torch

    from bittrainer.amp import amp_label, resolve_amp_dtype
    from bittrainer.autobatch import RunRefused
    from bittrainer.cloud_export import (
        SampleRef,
        build_scoped_export,
        verify_scoped_cache,
    )
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.group_task import GroupTask
    from bittrainer.generic.tasks.head_only_task import HeadOnlyTask
    from bittrainer.group_trainer import _get_dtype

    if pause_after_steps and epochs < 2:
        raise ValueError("--pause-after-steps needs at least two epochs to resume into")
    root = Path(root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    home, pod = root / "home", root / "pod"
    group = home / GROUP
    settings = dict(
        size=size, bucket_size=bucket_size, epochs=epochs, batch_size=batch_size,
        amp=amp, device=device, compile=compile, stock_grn=stock_grn, init=init,
        trunk=trunk, workers=workers,
    )  # fmt: skip
    counts = make_group_folder(
        group,
        per_class=images_per_class,
        val_per_class=max(2, images_per_class // 3),
        image_size=bucket_size,
        seed=seed,
    )
    checks: dict = {}
    head: dict = {}
    frames: list[dict] = []
    result: dict | None = None
    error = refused = None
    paused: dict | None = None
    task = None
    on_gpu = device == "cuda" and torch.cuda.is_available()
    try:
        # 1. Head-only at home, on the drawn images.
        head_task = HeadOnlyTask(
            build_config(
                group, home / "checkpoints", cache_dir=home / "cache",
                embedding_dir=home / "embeddings", from_scratch=True, **settings,
            )
        )  # fmt: skip
        began = time.perf_counter()
        head_result = GenericTrainer().run(head_task)
        head = {
            "seconds": round(time.perf_counter() - began, 1),
            "val_macro_f1": head_result.get("best_val_macro_f1"),
            "lean_grn": head_task.lean_grn,
        }
        incumbent = head_result.get("checkpoint_path")
        checks["head_only_completed"] = (
            not head_result.get("cancelled") and bool(incumbent) and Path(incumbent).is_file()
        )
        if not checks["head_only_completed"]:
            raise RuntimeError("the head-only run left no model")

        # 2. The scoped export of the group's cached tensors, as the cloud ships it.
        refs = [
            SampleRef(
                path=os.path.normpath(sample["path"]),
                split=split,
                label=sample.get("label"),
                concept_name=sample.get("concept_name", ""),
            )
            for split, dataset in (("train", head_task.train_ds), ("val", head_task.val_ds))
            for sample in dataset.samples
        ]
        manifest = build_scoped_export(
            global_cache_dir=home / "cache", export_dir=pod / "export", samples=refs,
            modeltype=head_task.config.modeltype, num_classes=len(CLASSES),
            class_names=list(CLASSES), group_name=GROUP, safe_name=GROUP,
        )  # fmt: skip
        verified, _key, message = verify_scoped_cache(pod / "export" / "cache", manifest)
        checks["export_verified"] = verified
        if not verified:
            raise RuntimeError(f"the export did not verify: {message}")
        # The pod never has the images: delete them before the update.
        shutil.rmtree(group)
        checks["sources_removed"] = not group.exists()
        (pod / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(incumbent, pod / "checkpoints" / "best.pt")
        (pod / GROUP).mkdir(parents=True, exist_ok=True)

        # 3. The sourceless full update of the head-only model.
        def _update_config(**extra):
            config = build_config(
                pod / GROUP, pod / "checkpoints", cache_dir=pod / "export" / "cache",
                embedding_dir=pod / "embeddings", sourceless=True, preflight=preflight,
                backup_dir=pod / "backups", **settings, **extra,
            )  # fmt: skip
            # On the config, not the run: the loop announces its back-pass mode
            # through the config's callback.
            config.progress_callback = frames.append
            return config

        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        pause = threading.Event() if pause_after_steps else None
        task = _pausing_task(_update_config(), pause, pause_after_steps)
        began = time.perf_counter()
        result = GenericTrainer().run(task, pause_event=pause)
        if pause is not None:
            paused = result
            task = GroupTask(_update_config(resume_from=pod / "backups"))
            result = GenericTrainer().run(task)
        update_seconds = round(time.perf_counter() - began, 1)
    except RunRefused as exc:
        refused = str(exc)
        update_seconds = None
    except Exception as exc:  # noqa: BLE001 - the report is the product of a proof run
        error = f"{type(exc).__name__}: {exc}"
        update_seconds = None

    numbers = update_numbers(frames)
    # An epoch, or a step of one (the warm-up's head probe is not a step).
    trained = any(
        f.get("type") == "epoch_complete"
        or (f.get("stage") == "training" and "throughput" in f)
        for f in frames
    )
    if expect_refusal:
        checks["refused"] = refused is not None
        checks["nothing_trained"] = not trained
        checks["no_backup_left"] = _no_backup(pod / "backups")
    elif refused is not None:
        error = f"RunRefused: {refused}"
    if not expect_refusal and "sources_removed" in checks:
        completed = result is not None and not result.get("paused")
        losses = [row["train_loss"] for row in numbers["epochs"]]
        checks["update_completed"] = completed
        checks["loss_finite"] = bool(losses) and all(
            loss is not None and math.isfinite(loss) for loss in losses
        )
        checks["checkpoint_finite"] = completed and _checkpoint_finite(
            result.get("checkpoint_path")
        )
        if pause_after_steps:
            checks["paused_with_a_backup"] = bool(
                paused and paused.get("paused") and paused.get("backup_path")
            )
            checks["resumed"] = numbers["resumed"]
        checks["score_reached"] = (
            None
            if min_score is None
            else bool(completed and (result.get("best_val_macro_f1") or 0.0) >= min_score)
        )

    dtype = resolve_amp_dtype(_get_dtype(_AMP_DTYPES[amp]), torch.device(device))
    gpu_name = capability = peak_vram_mb = None
    if on_gpu:
        props = torch.cuda.get_device_properties(torch.device(device))
        gpu_name, capability = props.name, f"{props.major}.{props.minor}"
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 2**20)
    grad_scaler = task.scaler is not None if task is not None else None
    lean_grn = task.lean_grn if task is not None else None
    return {
        "torch": torch.__version__,
        "device": device,
        "gpu_name": gpu_name,
        "compute_capability": capability,
        "size": size,
        "bucket_size": bucket_size,
        "images_per_class": images_per_class,
        "train_images": counts["train"],
        "val_images": counts["val"],
        "batch_size": batch_size or "auto",
        "batch_size_used": numbers["batch_size_used"],
        "workers": workers,
        "compile_requested": compile,
        "compiled": numbers["compiled"],
        "amp_dtype": amp_label(dtype != torch.float32, dtype),
        "grad_scaler": grad_scaler,
        "lean_grn": lean_grn,
        "fused_back_pass": numbers["fused_back_pass"],
        "skipped_steps": getattr(task, "skipped_steps", None) if grad_scaler else None,
        "epochs": numbers["epochs"],
        "best_images_per_second": numbers["best_images_per_second"],
        "peak_vram_mb": peak_vram_mb,
        "preflight_mb": numbers["preflight_mb"],
        "head_only": head,
        "update_seconds": update_seconds,
        "val_macro_f1": (result or {}).get("best_val_macro_f1"),
        "min_score": min_score,
        "paused": bool(paused and paused.get("paused")),
        "resumed": numbers["resumed"],
        "refused": refused,
        "seconds": round(time.perf_counter() - started, 1),
        "error": error,
        "checks": checks,
        "passed": error is None
        and bool(checks)
        and all(value for value in checks.values() if value is not None),
    }


def _pausing_task(config, pause: threading.Event | None, after: int):
    """A group task that asks to pause at its *after*-th optimizer step, the
    way an operator would, landing exactly at that step's boundary (progress
    frames are throttled, so they cannot place it). Without *pause* it is an
    ordinary group task."""
    from bittrainer.generic.tasks.group_task import GroupTask

    class PausingTask(GroupTask):
        def train_epoch(self, ctx, model, optimizer, train_loader, *, step_callback,
                        boundary_hook, start_batch):  # fmt: skip
            def _hook(num_batches: int):
                self.steps_seen += 1
                if self.steps_seen >= after:
                    pause.set()
                return boundary_hook(num_batches)

            return super().train_epoch(
                ctx, model, optimizer, train_loader, step_callback=step_callback,
                boundary_hook=_hook if pause is not None else boundary_hook,
                start_batch=start_batch,
            )  # fmt: skip

    task = PausingTask(config)
    task.steps_seen = 0
    return task


# --------------------------------------------------------------------------- #
# Command line                                                                #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bittrainer.tools.group_proof",
        description=(
            "Train a group head-only, export it, and update it sourceless, on "
            "generated images; report whether it held together and how fast."
        ),
    )
    parser.add_argument("--size", default="atto", help="ConvNeXt V2 size (atto ... large)")
    parser.add_argument(
        "--bucket-size", type=int, default=512,
        help="training resolution: the square bucket's side (512 = the default table)",
    )  # fmt: skip
    parser.add_argument("--images-per-class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--batch-size", type=int, default=0, help="0 = the automatic batch size"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--amp", default="auto", choices=sorted(_AMP_DTYPES))
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="regional torch.compile of the update (on by default, as Engine runs it)",
    )  # fmt: skip
    parser.add_argument(
        "--stock-grn", action="store_true",
        help="timm's own GRN layer instead of the lean one (fp16 cards only)",
    )  # fmt: skip
    parser.add_argument("--init", default="random", choices=["random", "timm"])
    parser.add_argument("--trunk", default=None, help="a local trunk .safetensors to start from")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--pause-after-steps", type=int, default=0,
        help="pause the update at this optimizer step, then resume it from its backup",
    )  # fmt: skip
    parser.add_argument(
        "--preflight", action=argparse.BooleanOptionalAction, default=None,
        help="force (or skip) the one-step preflight; default: on an fp16 card only",
    )  # fmt: skip
    parser.add_argument(
        "--expect-refusal", action="store_true",
        help="the update must be refused before it trains, leaving no backup",
    )  # fmt: skip
    parser.add_argument(
        "--min-score", type=float, default=-1.0,
        help="validation macro-F1 the update must reach; negative skips the check",
    )  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workdir", default=None,
        help="keep the images, export and models here (default: temp, deleted)",
    )  # fmt: skip
    args = parser.parse_args(argv)

    options = {
        "size": args.size,
        "bucket_size": args.bucket_size,
        "images_per_class": args.images_per_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": None if args.device == "auto" else args.device,
        "amp": args.amp,
        "compile": args.compile,
        "stock_grn": args.stock_grn,
        "init": args.init,
        "trunk": args.trunk,
        "workers": args.workers,
        "pause_after_steps": args.pause_after_steps,
        "preflight": args.preflight,
        "expect_refusal": args.expect_refusal,
        "min_score": None if args.min_score < 0 else args.min_score,
        "seed": args.seed,
    }
    if args.workdir:
        report = run_proof(Path(args.workdir), **options)
    else:
        with tempfile.TemporaryDirectory(prefix="group_proof_") as workdir:
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
