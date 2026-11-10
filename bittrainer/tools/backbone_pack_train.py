"""Train the backbone from an exported pack, headless (Bitcrush ISSUE-1655).

For a rented GPU with no Engine on it. Point it at a pack (see
``bittrainer.backbone_pack``) and an output folder::

    python -m bittrainer.tools.backbone_pack_train --pack /data/pack --out /data/run1 \
        --size base --image-size 224 --finetune-image-size 384 --finetune-epochs 5

    # a short trial on part of the pack first
    python -m bittrainer.tools.backbone_pack_train --pack /data/pack --out /data/trial \
        --limit 5000 --epochs 1

Everything the run produces goes into ``--out``:

    progress.jsonl                 every progress frame, appended
    backbone_candidate.safetensors the trained backbone and its heads
    result.json                    scores of the exported epoch
    backups/                       training state, to resume from

Stopping and resuming. The run backs itself up at every epoch end and every
``--backup-every`` steps. To control a detached run, create an empty file in
``--out``:

    PAUSE   back up now and exit (exit code 3)
    STOP    finish early: export the best epoch so far and exit

Ctrl-C and a terminate signal pause the same way. Running the same command
again resumes from the newest backup. If the backups belong to a different run
(another size, epoch count, resolution or pack) the command refuses rather than
train over them; ``--fresh`` discards them.

Exit codes: 0 finished, 1 failed, 2 refused, 3 paused.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import threading
import traceback
from pathlib import Path

from bittrainer.backbone_pack import PackError, pack_records
from bittrainer.convnextv2_catalog import get_convnextv2_pretrained_source

_AMP_CHOICES = {"auto": "auto", "fp16": "float16", "bf16": "bfloat16"}
_EXIT_CODES = {"finished": 0, "failed": 1, "refused": 2, "paused": 3}


def _log(message: str) -> None:
    print(f"[train] {message}", flush=True)


class Control:
    """Pause / stop switches for a run, driven by code, by signals, or by an
    empty ``PAUSE`` / ``STOP`` file dropped into the output folder."""

    def __init__(self, out: Path) -> None:
        self.out = Path(out)
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()

    def pause(self) -> None:
        self.pause_event.set()

    def stop(self) -> None:
        self.stop_event.set()

    def poll(self) -> None:
        """Honour a control file if one has appeared."""
        if (self.out / "PAUSE").exists():
            self.pause()
        if (self.out / "STOP").exists():
            self.stop()

    def clear_files(self) -> None:
        """Remove control files left by an earlier run, so it can start."""
        for name in ("PAUSE", "STOP"):
            (self.out / name).unlink(missing_ok=True)


def _backup_files(out: Path) -> list[Path]:
    return sorted((Path(out) / "backups").glob("backup_*.pt"))


def build_request(
    pack: Path,
    out: Path,
    *,
    size: str = "base",
    image_size: int = 224,
    epochs: int = 30,
    batch_size: int = 64,
    device: str | None = None,
    amp: str = "auto",
    init: str = "timm",
    init_checkpoint: str | None = None,
    workers: int = 4,
    channels_last: bool = False,
    finetune_image_size: int = 0,
    finetune_epochs: int = 0,
    soft_negatives: bool = False,
    validation_split: float | None = None,
    backup_every_steps: int = 2000,
    progress_every_steps: int = 50,
    max_steps: int | None = None,
    limit: int | None = None,
    fresh: bool = False,
    extra: dict | None = None,
) -> tuple[dict, dict]:
    """The ``run_backbone_training`` request for a pack, and what the pack held.

    Every training default the Backbone Builder ships with is left alone; only
    what is named here is set. ``extra`` is merged into ``training_config``
    last, for one-off experiments. ``fresh`` deletes old backups first.
    """
    pack, out = Path(pack), Path(out)
    records, counts = pack_records(pack, limit=limit)
    backups = out / "backups"
    if fresh:
        for stale in _backup_files(out):
            stale.unlink()

    config: dict = {
        "image_size": int(image_size),
        "batch_size": int(batch_size),  # 0 asks the trainer to probe for one
        "epochs": int(epochs),
        "dataloader_workers": int(workers),
        "channels_last": bool(channels_last),
        "finetune_image_size": int(finetune_image_size),
        "finetune_epochs": int(finetune_epochs),
        "backup_dir": str(backups),
        # Always offered: an empty folder starts fresh, a backup resumes.
        "resume_from": str(backups),
        "backup_every_steps": int(backup_every_steps),
        "progress_every_steps": int(progress_every_steps),
    }
    if device:
        config["device"] = device
    if amp == "off":
        config["use_amp"] = False
    else:
        config["use_amp"] = True
        config["amp_dtype"] = _AMP_CHOICES[amp]
    if soft_negatives:
        config["label_policy"] = {
            "mode": "soft_implicit_negative",
            "implicit_negative_value": 0.1,
        }
    if validation_split is not None:
        config["validation_split"] = float(validation_split)
    if max_steps is not None:
        config["max_steps"] = int(max_steps)
    config.update(extra or {})

    request: dict = {
        "run_id": out.name,
        "family_name": "CrushVision",
        "architecture": "convnextv2_compatible",
        "size_alias": size,
        "display_size": size,
        "convnextv2_size": size,
        "candidate_checkpoint_path": str(out / "backbone_candidate.safetensors"),
        "records": records,
        "training_config": config,
        "external_pretrained_used": False,
        "license_provenance": "locally_trained",
    }
    if init_checkpoint:
        request["backbone_init"] = {
            "source": "local_candidate",
            "checkpoint_path": str(init_checkpoint),
        }
        request["initialization_source"] = "local_candidate"
    elif init == "timm":
        # The weights this run starts from are named, with their licence, in
        # everything it exports.
        source = get_convnextv2_pretrained_source(size)
        request["backbone_init"] = {"source": "timm_pretrained"}
        request.update(
            initialization_source="timm_pretrained",
            license_provenance=source.model_id,
            pretrained_model_id=source.model_id,
            pretrained_license=source.license,
            pretrained_source_url=source.source_url,
            non_commercial_only=source.non_commercial_only,
            external_pretrained_used=True,
        )
    else:
        request["backbone_init"] = {"source": "random_init", "checkpoint_path": None}
        request["initialization_source"] = "random_init"
    return request, counts


def _backups_fit(request: dict) -> bool:
    """Whether the newest backup in the run's folder belongs to this run.

    Asked of the trainer itself, before anything trains: a run that starts over
    would rotate the old backups out within two saves.
    """
    from bittrainer.generic.tasks.backbone_task import BackboneTask

    task = BackboneTask(request)
    context = task.make_context(None, None, None, None)
    task.fingerprint_init(context)
    return context.resume_state is not None


def _jsonable(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _line(frame: dict) -> str | None:
    """One console line for a progress frame, or None for the quiet ones."""
    kind, stage = frame.get("type"), frame.get("stage")
    if stage == "preparing" and "train_samples" in frame:
        scaler = " with a GradScaler" if frame.get("grad_scaler") else ""
        return (
            f"{frame['train_samples']:,} train / {frame['val_samples']:,} validation images, "
            f"{frame.get('concepts', 0):,} yes/no heads, {frame.get('groups', 0):,} groups; "
            f"{frame.get('device')}, precision {frame.get('amp_dtype')}{scaler}"
        )
    if stage == "training_step":
        speed = frame.get("images_per_second") or 0.0
        return (
            f"epoch {frame['epoch']}/{frame['epochs']} batch {frame['batch']:,}/"
            f"{frame['batches']:,} loss {frame['loss']:.4f} {speed:.1f} img/s"
        )
    if stage == "training" and "loss" in frame:
        speed = frame.get("images_per_second") or 0.0
        best = frame.get("best_score")
        best_text = "none yet" if best is None else f"{best:.4f}"
        return (
            f"epoch {frame['epoch']}/{frame['epochs']} done: loss {frame['loss']:.4f}, "
            f"validation {frame['validation_score']:.4f}, best {best_text} "
            f"(epoch {frame.get('best_epoch')}), {speed:.1f} img/s"
        )
    if kind == "backup_complete":
        return (
            f"backup saved at step {frame.get('global_step')} ({frame.get('reason')})"
        )
    if kind == "training_resumed":
        return f"resumed at epoch {int(frame['epoch']) + 1}, step {frame.get('global_step')}"
    if kind == "training_paused":
        return f"paused at step {frame.get('global_step')}"
    if kind in ("graceful_stop", "stop_now"):
        return "stopping early; exporting the best epoch so far"
    if stage in ("saving", "training") and frame.get("status_text"):
        return str(frame["status_text"])
    return None


def train_from_pack(
    pack: Path,
    out: Path,
    *,
    on_frame=None,
    allow_missing: bool = False,
    install_signal_handlers: bool = False,
    poll_seconds: float = 2.0,
    **options,
) -> dict:
    """Train from *pack* into *out*. Returns ``{"status": ...}``: ``finished``,
    ``paused``, ``refused`` (with a ``reason``) or ``failed`` (with a ``reason``).

    ``on_frame(frame, control)`` sees every progress frame and may pause or stop
    the run through *control*.
    """
    pack, out = Path(pack), Path(out)
    try:
        request, counts = build_request(pack, out, **options)
    except PackError as exc:
        return {"status": "refused", "reason": str(exc)}
    total = counts["images"] + counts["missing_files"]
    if counts["missing_files"] and not allow_missing:
        return {
            "status": "refused",
            "reason": (
                f"{counts['missing_files']:,} of {total:,} images are missing from the pack "
                f"folder (an unfinished upload?). Finish the upload, or pass --allow-missing."
            ),
        }
    if not counts["images"]:
        return {"status": "refused", "reason": "the pack has no images to train on"}
    if _backup_files(out) and not _backups_fit(request):
        return {
            "status": "refused",
            "reason": (
                f"the backups in {out / 'backups'} belong to a different run (another size, "
                f"epoch count, resolution or pack). Use a new --out folder, or pass --fresh "
                f"to discard them."
            ),
        }

    import torch

    from bittrainer.backbone_trainer import run_backbone_training

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = (
            True  # fixed input sizes: let cuDNN pick kernels
        )

    out.mkdir(parents=True, exist_ok=True)
    control = Control(out)
    control.clear_files()
    if install_signal_handlers:
        for name in ("SIGINT", "SIGTERM"):
            signal.signal(getattr(signal, name), lambda *_args: control.pause())

    watching = threading.Event()

    def _watch() -> None:
        while not watching.wait(poll_seconds):
            control.poll()

    watcher = threading.Thread(target=_watch, name="pack-train-control", daemon=True)
    watcher.start()
    _log(f"{counts['images']:,} images from the pack; output in {out}")
    result: dict | None = None
    failure: str | None = None
    with (out / "progress.jsonl").open("a", encoding="utf-8") as progress:

        def _frame(frame: dict) -> None:
            progress.write(json.dumps(frame, default=_jsonable) + "\n")
            progress.flush()
            line = _line(frame)
            if line:
                _log(line)
            if on_frame is not None:
                on_frame(frame, control)

        try:
            result = asyncio.run(
                run_backbone_training(
                    request,
                    progress_callback=_frame,
                    pause_event=control.pause_event,
                    stop_event=control.stop_event,
                )
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            watching.set()
            watcher.join(timeout=poll_seconds + 1)

    if failure is not None:
        return {"status": "failed", "reason": failure}
    if result.get("paused"):
        return {"status": "paused", "backup": result.get("backup_path")}
    (out / "result.json").write_text(
        json.dumps(result, indent=1, default=_jsonable), encoding="utf-8"
    )
    return {
        "status": "finished",
        "validation_score": result.get("validation_score"),
        "best_epoch": result.get("best_epoch"),
        "epochs_completed": result.get("epochs_completed"),
    }


def _setting(text: str) -> tuple[str, object]:
    """``key=value`` for ``--set``; the value is JSON when it parses, else text."""
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    try:
        return key, json.loads(raw)
    except ValueError:
        return key, raw


def _batch_size(text: str) -> int:
    return 0 if text.lower() == "auto" else int(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bittrainer.tools.backbone_pack_train",
        description="Train the backbone from an exported pack, headless.",
    )
    parser.add_argument("--pack", required=True, help="the pack folder")
    parser.add_argument("--out", required=True, help="where the run writes everything")
    parser.add_argument(
        "--size", default="base", help="ConvNeXt V2 size (default base)"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--finetune-image-size", type=int, default=0, help="size of the tail"
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=0, help="last N epochs at it"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size", type=_batch_size, default=64, help="a number, or auto"
    )
    parser.add_argument("--workers", type=int, default=4, help="image loading workers")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--amp", default="auto", choices=["auto", "fp16", "bf16", "off"]
    )
    parser.add_argument(
        "--device", default=None, help="cuda or cpu (default: cuda if present)"
    )
    parser.add_argument(
        "--init",
        default="timm",
        choices=["timm", "random"],
        help="timm = the published ConvNeXt V2 weights (CC-BY-NC-4.0); random = from nothing",
    )
    parser.add_argument(
        "--init-checkpoint", default=None, help="start from a local trunk"
    )
    parser.add_argument(
        "--soft-negatives",
        action="store_true",
        help="unlabelled images count as a soft no (0.1) for a head, under its cap",
    )
    parser.add_argument("--validation-split", type=float, default=None)
    parser.add_argument(
        "--backup-every", type=int, default=2000, help="steps between backups"
    )
    parser.add_argument(
        "--progress-every", type=int, default=50, help="steps between heartbeats"
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="train on the first N images"
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--fresh", action="store_true", help="discard old backups in --out"
    )
    parser.add_argument(
        "--set",
        dest="extra",
        action="append",
        type=_setting,
        default=[],
        metavar="KEY=VALUE",
        help="any other training setting, e.g. --set neg_pos_ratio=3",
    )
    args = parser.parse_args(argv)

    summary = train_from_pack(
        Path(args.pack),
        Path(args.out),
        allow_missing=args.allow_missing,
        install_signal_handlers=True,
        size=args.size,
        image_size=args.image_size,
        finetune_image_size=args.finetune_image_size,
        finetune_epochs=args.finetune_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        channels_last=args.channels_last,
        amp=args.amp,
        device=args.device,
        init=args.init,
        init_checkpoint=args.init_checkpoint,
        soft_negatives=args.soft_negatives,
        validation_split=args.validation_split,
        backup_every_steps=args.backup_every,
        progress_every_steps=args.progress_every,
        max_steps=args.max_steps,
        limit=args.limit,
        fresh=args.fresh,
        extra=dict(args.extra),
    )
    status = summary["status"]
    if status == "finished":
        _log(
            f"finished: validation {summary.get('validation_score')}, best epoch "
            f"{summary.get('best_epoch')} of {summary.get('epochs_completed')}"
        )
    elif status == "paused":
        _log("paused: state backed up. Run the same command again to resume.")
    else:
        _log(f"{status.upper()}: {summary.get('reason')}")
    return _EXIT_CODES.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
