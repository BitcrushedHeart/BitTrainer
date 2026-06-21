"""A/B benchmark: original recipe vs Phase-2 augmentations.

Trains the same group from scratch twice on identical data and seed — once with
the Phase-2 levers OFF (reproducing the original recipe: no EMA, no SWA, no
MixUp, replication balancing) and once with them ON (EMA + SWA + MixUp/CutMix +
auto class-reweighting) — and prints the QWK / macro-F1 deltas.

Both arms run on the SAME code, so the comparison isolates the augmentation
effect rather than confounding it with unrelated changes. Decode (Phase 1) is
identical in both arms and therefore controlled for.

Default target is the "Inner Labia" group (id 20, ordinal, 6 classes incl.
__none__, class index 2 = "Innie Labia"). Override via env vars:
    GROUP_FOLDER, GROUP_NAME, EPOCHS, DEVICE.

Run:  python benchmarks/ab_phase2.py
"""

from __future__ import annotations

import os
import random
import tempfile
import time

import numpy as np
import torch

from bittrainer.group_trainer import GroupTrainConfig, run_group_training

# Fixed for both arms so the comparison isolates the augmentation effect:
#  - BATCH_SIZE skips the auto batch-size probe (which OOM-churns near the VRAM
#    cap and dominates wall-clock for a 3-epoch run);
#  - a shared embedding-cache dir means the head-probe warmup's backbone-feature
#    pass is built once and reused by the second arm.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
HEAD_MAX_EPOCHS = int(os.environ.get("HEAD_MAX_EPOCHS", "12"))
# Stable (not per-run) so re-runs reuse the backbone-feature cache and skip the
# ~70s warmup embedding build.
_EMB_CACHE_DIR = os.path.join(tempfile.gettempdir(), "ab_phase2_embcache")
os.makedirs(_EMB_CACHE_DIR, exist_ok=True)
_RESULT_JSON = os.path.join(tempfile.gettempdir(), "ab_phase2_result.json")

GROUP_FOLDER = os.environ.get("GROUP_FOLDER", r"F:\groups\Labia")
GROUP_NAME = os.environ.get("GROUP_NAME", "Inner Labia")
CLASS_NAMES = [
    "__none__", "Full Innie Labia", "Innie Labia",
    "Even Labia", "Outie Labia", "Protruding Labia",
]
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = int(os.environ.get("EPOCHS", "6"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = os.environ.get(
    "CACHE_DIR",
    r"F:\Projects\Bitcrush\Bitcrush Suite\apps\engine\data\training_cache\image",
)
SEED = 1234


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _base_config(checkpoint_dir: str) -> dict:
    return dict(
        group_folder=GROUP_FOLDER,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        group_name=GROUP_NAME,
        max_epochs=EPOCHS,
        patience=EPOCHS,            # never early-stop in a 3-epoch benchmark
        from_scratch=True,
        backbone_variant="nano",
        ordinal=True,
        ordinal_sigma=0.9,
        label_smoothing=0.1,
        validation_metric="qwk",
        multi_label=False,
        oversample_none=False,
        skin_normalise=False,
        checkpoint_dir=checkpoint_dir,
        best_model_name="ab_candidate.pt",
        cache_dir=CACHE_DIR,
        use_cache=True,
        device=DEVICE,
        use_compile=False,          # short run — skip compile overhead
        auto_label_softness=False,  # pin sigma so the sweep can't confound the A/B
        batch_size=BATCH_SIZE,      # fixed -> skip the auto batch-size probe
        embedding_cache_dir=_EMB_CACHE_DIR,  # shared backbone-feature cache across arms
        head_max_epochs=HEAD_MAX_EPOCHS,     # cap the head-probe warmup (same for both arms)
    )


# Original recipe: every Phase-2 lever off.
BASELINE_FLAGS = dict(
    use_ema=False,
    use_swa=False,
    use_mixup=False,
    class_balance_mode="resample",
    use_focal=False,
)

# Phase-2 recipe: the new defaults.
NEW_FLAGS = dict(
    use_ema=True,
    ema_decay=0.999,
    use_swa=True,
    swa_start_frac=0.5,            # average weights over the back half of training
    use_mixup=True,
    class_balance_mode="auto",
    use_focal=False,
)


def _make_progress_printer(label: str):
    """Print concise, flushed stage/epoch lines so the run is observable when
    stdout is redirected to a file (block-buffered otherwise)."""
    t0 = time.monotonic()
    last_stage = [None]

    def cb(msg: dict) -> None:
        et = f"{time.monotonic() - t0:6.0f}s"
        mtype = msg.get("type")
        if mtype == "epoch_complete":
            print(
                f"[{label} {et}] epoch {msg.get('epoch')}/{msg.get('max_epochs')} "
                f"| val_macro_f1={msg.get('val_macro_f1')} val_qwk={msg.get('val_qwk')}",
                flush=True,
            )
        elif mtype in ("training_complete", "stop_now", "graceful_stop"):
            print(f"[{label} {et}] {mtype}", flush=True)
        else:
            stage = msg.get("stage") or mtype
            if stage and stage != last_stage[0]:
                last_stage[0] = stage
                txt = msg.get("status_text", "")
                print(f"[{label} {et}] » {stage}: {txt}"[:140], flush=True)

    return cb


def _run(label: str, flags: dict) -> dict:
    print(f"\n{'=' * 70}\n[{label}] starting — {EPOCHS} epochs, from scratch\n{'=' * 70}", flush=True)
    _seed_everything(SEED)
    with tempfile.TemporaryDirectory(prefix=f"ab_{label}_") as ckpt_dir:
        cfg = {**_base_config(ckpt_dir), **flags}
        config = GroupTrainConfig(**cfg)
        t0 = time.monotonic()
        result = run_group_training(config, progress_callback=_make_progress_printer(label))
        result["_elapsed_s"] = round(time.monotonic() - t0, 1)
    return result


def _metrics(r: dict) -> dict:
    return {
        "qwk": r.get("qwk"),
        "best_val_qwk": r.get("best_val_qwk"),
        "macro_f1": r.get("final_val_macro_f1"),
        "ordinal_mae": r.get("ordinal_mae"),
        "adjacent_acc": r.get("adjacent_accuracy"),
        "sel_score": r.get("selected_validation_score"),
        "ordinal_decode": r.get("ordinal_decode"),
        "elapsed_s": r.get("_elapsed_s"),
    }


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def main() -> None:
    import json

    baseline = _metrics(_run("baseline", BASELINE_FLAGS))
    new = _metrics(_run("phase2", NEW_FLAGS))

    # Persist first, so a console-encoding hiccup can never lose the numbers.
    with open(_RESULT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"group": GROUP_NAME, "epochs": EPOCHS, "seed": SEED,
                   "baseline": baseline, "phase2": new}, fh, indent=2)

    print(f"\n{'=' * 70}\nA/B RESULT - {GROUP_NAME} ({EPOCHS} epochs, seed {SEED})\n{'=' * 70}")
    header = f"{'metric':<16}{'baseline':>14}{'phase2':>14}{'delta':>14}"
    print(header)
    print("-" * len(header))
    for key in ["qwk", "best_val_qwk", "macro_f1", "adjacent_acc", "ordinal_mae", "sel_score"]:
        b, n = baseline.get(key), new.get(key)
        delta = (n - b) if isinstance(b, (int, float)) and isinstance(n, (int, float)) else None
        flag = ""
        if delta is not None:
            # For ordinal_mae lower is better; everything else higher is better.
            better = (delta < 0) if key == "ordinal_mae" else (delta > 0)
            flag = "  [+]" if delta and better else ("  [-]" if delta else "")
        print(f"{key:<16}{_fmt(b):>14}{_fmt(n):>14}{_fmt(delta):>14}{flag}")
    print("-" * len(header))
    print(f"{'decode':<16}{str(baseline['ordinal_decode']):>14}{str(new['ordinal_decode']):>14}")
    print(f"{'train secs':<16}{_fmt(baseline['elapsed_s']):>14}{_fmt(new['elapsed_s']):>14}")
    print(f"\nResults JSON: {_RESULT_JSON}")
    print(f"\nNote: {EPOCHS} epochs is still a short benchmark - EMA/SWA need more epochs "
          "to fully engage; MixUp + reweighting trade early-epoch fit for regularisation. "
          "Run with EPOCHS=15+ for the fuller picture.")


if __name__ == "__main__":
    main()
