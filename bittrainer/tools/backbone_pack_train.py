"""Train the backbone from an exported pack, headless (Bitcrush ISSUE-1655).

For a rented GPU with no Engine on it. Point it at a pack (see
``bittrainer.backbone_pack``) and an output folder::

    python -m bittrainer.tools.backbone_pack_train --pack /data/pack --out /data/run1 \
        --size base --image-size 224 --batch-size 64 \
        --finetune-image-size 384 --finetune-epochs 5 --finetune-batch-size 16

    # a short trial on part of the pack first
    python -m bittrainer.tools.backbone_pack_train --pack /data/pack --out /data/trial \
        --limit 5000 --epochs 1

Everything the run produces goes into ``--out``:

    progress.jsonl                 every progress frame, appended
    backbone_candidate.safetensors the trained backbone and its heads
    result.json                    scores of the exported epoch, and where its
                                   validation came from: "validation_source"
                                   is "home_split" when the pack (format 2)
                                   marks answers held for validation at home,
                                   which then never train, else "content_hash"
    backbone_candidate.val_scores.npz
                                   that epoch's raw validation scores, to work
                                   thresholds out again later
    backups/                       training state, to resume from

Stopping and resuming. The run backs itself up at every epoch end and every
``--backup-every`` steps. To control a detached run, create an empty file in
``--out``:

    PAUSE   back up now and exit (exit code 3)
    STOP    finish early: export the best epoch so far and exit

Ctrl-C and a terminate signal pause the same way. Running the same command
again resumes from the newest backup. A run that has finished (it left a
``result.json``) is left alone, so a service that starts this command at every
boot never trains a finished run again; ``--fresh`` does. If the backups belong
to a different run
(another size, epoch count, resolution or pack) the command refuses rather than
train over them; ``--fresh`` discards them. The batch sizes are not part of
that: a paused run can come back with another ``--batch-size`` or
``--finetune-batch-size`` (the epoch it stopped in starts again).

Before the first epoch, on a GPU, one real training step is run at each
resolution and batch size the run will use. If one does not fit the card the
run is refused there and then, naming which, instead of failing at the switch
to the tail a day in.

Every run starts from the Apache-2.0 ConvNeXt V1 weights (Bitcrush ISSUE-1721):
``--init apache`` (the default), or an ``--init-checkpoint`` / ``--teacher``
that descends from them. ``--init timm`` (CC-BY-NC-4.0) and ``--init random``
are refused, as is a trunk or teacher whose lineage does not reach them. The
candidate records its lineage (``lineage_json``, see ``bittrainer.lineage``),
and so does ``result.json``.

Exit codes: 0 finished, 1 failed, 2 refused, 3 paused.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import socket
import threading
import time
import traceback
from pathlib import Path

from bittrainer.backbone_pack import PackError, pack_geometry_sensitive, pack_records
from bittrainer.convnext_v1_init import write_v1_trunk
from bittrainer.lineage import (
    LineageError,
    extend,
    lineage_of,
    read_metadata,
    require_apache,
    trunk_hash,
)

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


def _header(path, what: str) -> dict:
    """The metadata of a model file named on the command line."""
    try:
        return read_metadata(path)
    except Exception as exc:  # missing, not a model file, unreadable
        raise LineageError(f"{what} cannot be read: {exc}") from exc


def _check_start(
    init: str, init_checkpoint: str | None, teacher: str | None
) -> dict | None:
    """Refuse any start that does not descend from the Apache ConvNeXt V1
    weights (Bitcrush ISSUE-1721), before anything is read or written. Returns
    the lineage of ``init_checkpoint`` when one is named."""
    start = None
    if init_checkpoint:
        what = f"the --init-checkpoint {Path(init_checkpoint).name}"
        meta = _header(init_checkpoint, what)
        require_apache(meta, what)
        start = lineage_of(meta)
    elif init != "apache":
        raise LineageError(
            f"--init {init} is refused: every model starts from the Apache ConvNeXt V1 "
            "weights (--init apache) or from a trunk that descends from them "
            "(--init-checkpoint), Bitcrush ISSUE-1721"
        )
    if teacher:
        what = f"the teacher {Path(teacher).name}"
        require_apache(_header(teacher, what), what)
    return start


def build_request(
    pack: Path,
    out: Path,
    *,
    size: str = "base",
    image_size: int = 224,
    epochs: int = 30,
    batch_size: int = 64,
    finetune_batch_size: int | None = None,
    device: str | None = None,
    amp: str = "auto",
    init: str = "apache",
    init_checkpoint: str | None = None,
    v1_weights: str | None = None,
    teacher: str | None = None,
    distill: dict | None = None,
    workers: int = 4,
    channels_last: bool = False,
    stock_grn: bool = False,
    finetune_image_size: int = 0,
    finetune_epochs: int = 0,
    soft_negatives: bool = False,
    hill_negatives: bool = False,
    reject_rate: float = 0.0,
    mask_common: float | None = None,
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

    ``init="apache"`` converts Apache-2.0 ConvNeXt V1 weights into the run's
    folder once (see ``bittrainer.convnext_v1_init``) and starts from that
    trunk; a resumed run reuses the file. ``init_checkpoint`` starts from a
    trunk that descends from them instead. Any other start raises
    :class:`~bittrainer.lineage.LineageError` (Bitcrush ISSUE-1721).
    ``teacher`` (a trained candidate with an Apache root) turns the run into a
    distillation of it, tuned by ``distill``.

    The request carries the lineage the candidate records: ``lineage_json``
    for a plain run; for a distillation ``lineage_start`` (its own start),
    which the distillation task extends from the teacher's lineage.

    ``stock_grn`` trains with timm's own GRN layer instead of the lean one the
    trainer uses by default (``bittrainer.lean_grn``): slower and heavier under
    half precision, there for comparison.
    """
    pack, out = Path(pack), Path(out)
    start = _check_start(init, init_checkpoint, teacher)
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
    if finetune_batch_size is not None:
        # The tail's own batch size (0 asks for a probe). Unset: as batch_size.
        config["finetune_batch_size"] = int(finetune_batch_size)
    if stock_grn:
        config["lean_grn"] = False
    if device:
        config["device"] = device
    if amp == "off":
        config["use_amp"] = False
    else:
        config["use_amp"] = True
        config["amp_dtype"] = _AMP_CHOICES[amp]
    # The three options act on unlabelled images taken as negatives, so asking
    # for any of them turns those on (see label_policy in backbone_trainer).
    if soft_negatives or hill_negatives or reject_rate or mask_common is not None:
        config["label_policy"] = {
            "mode": "soft_implicit_negative",
            "implicit_negative_value": 0.1,
        }
        if hill_negatives:
            config["label_policy"]["implicit_negative_loss"] = "hill"
        if reject_rate:
            config["label_policy"]["reject_rate"] = float(reject_rate)
        if mask_common is not None:
            config["label_policy"]["implicit_negative_max_prior"] = float(mask_common)
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
    geometry_sensitive = pack_geometry_sensitive(pack)
    if geometry_sensitive is not None:  # an empty list still says "looked, none"
        request["geometry_sensitive"] = geometry_sensitive
    from_trunk = None
    if init_checkpoint:
        request["backbone_init"] = {
            "source": "local_candidate",
            "checkpoint_path": str(init_checkpoint),
        }
        request["initialization_source"] = "local_candidate"
        from_trunk = trunk_hash(init_checkpoint)
    else:
        # Converted once into the run's folder; the note beside it carries
        # where it came from, so a resumed run does not download it again.
        trunk, note = out / "init_trunk.safetensors", out / "init_trunk.json"
        if trunk.is_file() and note.is_file():
            source = json.loads(note.read_text(encoding="utf-8"))
        else:
            source = write_v1_trunk(size, trunk, v1_weights)
            note.write_text(json.dumps(source, indent=1), encoding="utf-8")
        request["backbone_init"] = {
            "source": "local_candidate",
            "checkpoint_path": str(trunk),
        }
        request.update(
            initialization_source="convnext_v1_converted",
            license_provenance=source["model_id"],
            pretrained_model_id=source["model_id"],
            pretrained_license=source["license"],
            pretrained_source_url=source["source_url"],
            non_commercial_only=False,
            external_pretrained_used=True,
        )
        start = lineage_of(
            {"converted_from": source["model_id"], "license": source["license"]}
        )
        if start is None:
            raise LineageError(
                f"--init apache: {source['model_id']} ({source['license']}) is not "
                "Apache ConvNeXt V1 weights"
            )
    if teacher:
        request["distillation"] = {
            "teacher_checkpoint_path": str(teacher),
            **(distill or {}),
        }
        request["lineage_start"] = start
    else:
        step = {"kind": "backbone", "size": size, "from_trunk": from_trunk}
        request["lineage_json"] = extend(start, step)
    return request, counts


def _can_distil() -> bool:
    """Whether this build has the distillation task. Without it the trainer
    would ignore the teacher and quietly run an ordinary training."""
    name = "bittrainer.generic.tasks.backbone_distill_task"
    return importlib.util.find_spec(name) is not None


def _backups_fit(request: dict) -> bool:
    """Whether the newest backup in the run's folder belongs to this run.

    Asked of the trainer itself, before anything trains: a run that starts over
    would rotate the old backups out within two saves. The question goes to the
    task the run will use: a distillation's backups carry their own
    fingerprint, and the plain task would call them another run's.
    """
    from bittrainer.backbone_trainer import backbone_task_class

    task = backbone_task_class(request)(request)
    context = task.make_context(None, None, None, None)
    task.fingerprint_init(context)
    return context.resume_state is not None


def _jsonable(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _skips(frame: dict) -> str:
    """What the fp16 GradScaler is doing, for a console line. Nothing to say
    in bf16 or on a CPU, where there is no scaler."""
    scale = frame.get("grad_scale")
    if scale is None:
        return ""
    return f", {frame.get('skipped_steps', 0):,} skipped, scale {scale:g}"


def _line(frame: dict) -> str | None:
    """One console line for a progress frame, or None for the quiet ones."""
    kind, stage = frame.get("type"), frame.get("stage")
    if stage == "warning":
        return f"WARNING: {frame.get('status_text')}"
    if stage == "preparing" and "train_samples" in frame:
        scaler = " with a GradScaler" if frame.get("grad_scaler") else ""
        source = frame.get("validation_source")
        validation = ""
        if source == "home_split":
            validation = (
                f"; validation: home split, {frame.get('held_answers', 0):,} held "
                f"answers on {frame.get('held_images', 0):,} images"
            )
        elif source:
            validation = "; validation: by content hash"
        return (
            f"{frame['train_samples']:,} train / {frame['val_samples']:,} validation images, "
            f"{frame.get('concepts', 0):,} yes/no heads, {frame.get('groups', 0):,} groups; "
            f"{frame.get('device')}, precision {frame.get('amp_dtype')}{scaler}{validation}"
        )
    if stage == "training_step":
        speed = frame.get("images_per_second") or 0.0
        return (
            f"epoch {frame['epoch']}/{frame['epochs']} batch {frame['batch']:,}/"
            f"{frame['batches']:,} loss {frame['loss']:.4f} {speed:.1f} img/s"
            f"{_skips(frame)}"
        )
    if stage == "training" and "loss" in frame:
        speed = frame.get("images_per_second") or 0.0
        best = frame.get("best_score")
        best_text = "none yet" if best is None else f"{best:.4f}"
        return (
            f"epoch {frame['epoch']}/{frame['epochs']} done: loss {frame['loss']:.4f}, "
            f"validation {frame['validation_score']:.4f}, best {best_text} "
            f"(epoch {frame.get('best_epoch')}), {speed:.1f} img/s{_skips(frame)}"
        )
    if kind == "backup_complete":
        return (
            f"backup saved at step {frame.get('global_step')} ({frame.get('reason')})"
        )
    if kind == "training_resumed":
        sizes = ""
        if frame.get("batch_size"):
            sizes = f" ({frame.get('image_size')}px, batch {frame['batch_size']})"
        return (
            f"resumed at epoch {int(frame['epoch']) + 1}, step "
            f"{frame.get('global_step')}{sizes}"
        )
    if kind == "training_paused":
        return f"paused at step {frame.get('global_step')}"
    if kind in ("graceful_stop", "stop_now"):
        return "stopping early; exporting the best epoch so far"
    if kind == "autobatch":
        return (
            f"automatic batch size {frame.get('batch_size')} at "
            f"{frame.get('image_size')}px ({frame.get('phase')})"
        )
    if stage in ("saving", "training", "preflight") and frame.get("status_text"):
        return str(frame["status_text"])
    return None


# What backbone_status needs to know of a run's plan to say how long is left.
_PLAN_KEYS = (
    "epochs",
    "image_size",
    "batch_size",
    "finetune_image_size",
    "finetune_epochs",
    "finetune_batch_size",
)


def _write_status(out: Path, status: str, **fields) -> None:
    """Keep ``status.json`` current: it is how ``backbone_status`` tells a run
    that is running from one that paused, finished, failed or simply vanished."""
    path = Path(out) / "status.json"
    scratch = path.with_name("status.json.tmp")
    scratch.write_text(
        json.dumps({"status": status, **fields}, default=_jsonable), encoding="utf-8"
    )
    scratch.replace(path)


def _finished_result(out: Path) -> dict | None:
    """The result of the run that finished in *out*, or None when none has.
    ``result.json`` is only ever written by a run that finished."""
    path = Path(out) / "result.json"
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return result if isinstance(result, dict) else {}


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
    done = _finished_result(out)
    if done is not None and not options.get("fresh"):
        # Running the same command again resumes an interrupted run, and a
        # service does exactly that at every boot. A finished run has nothing
        # to resume: starting over would train it all again and write over its
        # finished model.
        return {
            "status": "finished",
            "already_finished": True,
            "validation_score": done.get("validation_score"),
            "best_epoch": done.get("best_epoch"),
            "epochs_completed": done.get("epochs_completed"),
        }
    teacher = options.get("teacher")
    if teacher and not _can_distil():
        return {
            "status": "refused",
            "reason": (
                "this build of BitTrainer cannot distil yet, so the teacher would be "
                "ignored and the run would be an ordinary training"
            ),
        }
    if teacher and not Path(teacher).is_file():
        return {
            "status": "refused",
            "reason": f"the teacher file was not found: {teacher}",
        }
    try:
        request, counts = build_request(pack, out, **options)
    except (PackError, ValueError, RuntimeError) as exc:
        # An unreadable pack, or starting weights that were refused or did not
        # convert cleanly.
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
    from bittrainer.generic.tasks.backbone_task import BackboneRunRefused

    if torch.cuda.is_available():
        # Fixed input sizes: let cuDNN pick its fastest kernels.
        torch.backends.cudnn.benchmark = True

    out.mkdir(parents=True, exist_ok=True)
    # Only --fresh brings a finished run this far. Its result goes first: cut
    # short, the new run must be resumed at the next start, not read as finished.
    (out / "result.json").unlink(missing_ok=True)
    control = Control(out)
    control.clear_files()
    plan = {key: request["training_config"].get(key) for key in _PLAN_KEYS}
    _write_status(
        out,
        "running",
        pid=os.getpid(),
        host=socket.gethostname(),
        started=round(time.time(), 1),
        plan=plan,
    )
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
    if "held_answers" in counts:  # a format 2 pack: answers held for validation
        _log(
            f"{counts['held_answers']:,} answers on {counts['held_images']:,} images "
            "are validation answers at home: they validate and never train"
        )
    result: dict | None = None
    failure: str | None = None
    refusal: str | None = None
    with (out / "progress.jsonl").open("a", encoding="utf-8") as progress:

        def _frame(frame: dict) -> None:
            # Stamped here, not in the trainer: only a detached run needs to be
            # asked later when it last made progress.
            frame = {**frame, "t": round(time.time(), 1)}
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
        except BackboneRunRefused as exc:
            # The trainer's own check before the first epoch (a training step
            # that does not fit the card): nothing trained, nothing broke.
            refusal = str(exc)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            watching.set()
            watcher.join(timeout=poll_seconds + 1)

    if refusal is not None:
        summary = {"status": "refused", "reason": refusal}
    elif failure is not None:
        summary = {"status": "failed", "reason": failure}
    elif result.get("paused"):
        summary = {"status": "paused", "backup": result.get("backup_path")}
    else:
        if torch.cuda.is_available():
            # What the run needed at its peak, to size the batch for the next one.
            result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        # Where the candidate came from, beside its scores (Bitcrush ISSUE-1721).
        try:
            candidate = read_metadata(result.get("candidate_checkpoint_path"))
            result["lineage_json"] = lineage_of(candidate)
        except (OSError, TypeError, ValueError):
            result["lineage_json"] = None
        (out / "result.json").write_text(
            json.dumps(result, indent=1, default=_jsonable), encoding="utf-8"
        )
        summary = {
            "status": "finished",
            "validation_score": result.get("validation_score"),
            "best_epoch": result.get("best_epoch"),
            "epochs_completed": result.get("epochs_completed"),
        }
    _write_status(out, plan=plan, ended=round(time.time(), 1), **summary)
    return summary


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
    parser.add_argument(
        "--finetune-batch-size",
        type=_batch_size,
        default=None,
        help="the tail's own batch size: a number, or auto (default: --batch-size)",
    )
    parser.add_argument("--workers", type=int, default=4, help="image loading workers")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--stock-grn",
        action="store_true",
        help="train with timm's own GRN layer (slower in half precision)",
    )
    parser.add_argument(
        "--amp", default="auto", choices=["auto", "fp16", "bf16", "off"]
    )
    parser.add_argument(
        "--device", default=None, help="cuda or cpu (default: cuda if present)"
    )
    parser.add_argument(
        "--init",
        default="apache",
        choices=["timm", "apache", "random"],
        help=(
            "apache (the default) = ConvNeXt V1 weights (Apache-2.0) converted into a "
            "V2 trunk. timm (the CC-BY-NC-4.0 ConvNeXt V2 weights) and random are "
            "refused: every model descends from the Apache weights"
        ),
    )
    parser.add_argument(
        "--v1-weights", default=None, help="with --init apache: another timm V1 weight"
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="start from a local trunk that descends from the Apache weights",
    )
    parser.add_argument(
        "--teacher", default=None, help="distil this trained candidate into the run"
    )
    parser.add_argument("--distill-temperature", type=float, default=None)
    parser.add_argument("--distill-logit-weight", type=float, default=None)
    parser.add_argument("--distill-feature-weight", type=float, default=None)
    parser.add_argument("--distill-supervised-weight", type=float, default=None)
    parser.add_argument(
        "--soft-negatives",
        action="store_true",
        help="unlabelled images count as a soft no (0.1) for a head, under its cap",
    )
    parser.add_argument(
        "--hill-negatives",
        action="store_true",
        help=(
            "soft negatives with the Hill term in place of the 0.1 target: it "
            "stops pushing an unlabelled image down once the model scores it high"
        ),
    )
    parser.add_argument(
        "--reject-rate",
        type=float,
        default=0.0,
        metavar="SHARE",
        help=(
            "soft negatives with large-loss rejection: from the second epoch each "
            "batch's highest-scoring unlabelled negatives are left out, this share "
            "more of them every epoch (0.01 = 1%% in epoch 2, 2%% in epoch 3, ...)"
        ),
    )
    parser.add_argument(
        "--mask-common",
        type=float,
        default=None,
        metavar="PRIOR",
        help=(
            "soft negatives for the rarer concepts only: a concept with a human yes "
            "on more than this share of the training images takes none (try 0.05)"
        ),
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
        finetune_batch_size=args.finetune_batch_size,
        workers=args.workers,
        channels_last=args.channels_last,
        stock_grn=args.stock_grn,
        amp=args.amp,
        device=args.device,
        init=args.init,
        init_checkpoint=args.init_checkpoint,
        v1_weights=args.v1_weights,
        teacher=args.teacher,
        distill={
            key: value
            for key, value in {
                "temperature": args.distill_temperature,
                "logit_weight": args.distill_logit_weight,
                "feature_weight": args.distill_feature_weight,
                "supervised_weight": args.distill_supervised_weight,
            }.items()
            if value is not None
        },
        soft_negatives=args.soft_negatives,
        hill_negatives=args.hill_negatives,
        reject_rate=args.reject_rate,
        mask_common=args.mask_common,
        validation_split=args.validation_split,
        backup_every_steps=args.backup_every,
        progress_every_steps=args.progress_every,
        max_steps=args.max_steps,
        limit=args.limit,
        fresh=args.fresh,
        extra=dict(args.extra),
    )
    status = summary["status"]
    if status == "finished" and summary.get("already_finished"):
        _log(
            f"already finished (validation {summary.get('validation_score')}, best epoch "
            f"{summary.get('best_epoch')} of {summary.get('epochs_completed')}): nothing was "
            f"trained. To train again, use a new --out folder or pass --fresh."
        )
    elif status == "finished":
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
