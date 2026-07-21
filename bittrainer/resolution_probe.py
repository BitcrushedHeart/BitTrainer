"""Training-resolution probe (Bitcrush ISSUE-0550).

Answers "what resolution should this group/concept train at?" empirically and
cheaply: sample the dataset (stratified, a few hundred images), embed the SAME
sample through the frozen init backbone at each candidate resolution, and
k-fold a tiny linear head on the cached vectors per resolution. Heads on
vectors fit in seconds, so the whole probe is dominated by one embedding pass
per resolution over the sample.

Statistical design: the sample AND the fold assignment are fixed once (seeded)
and shared by every resolution, and the per-fold head init is seeded per fold —
so each resolution is a PAIRED comparison against the baseline on identical
folds, which is dramatically tighter at n~300 than independent estimates.
Results are reported as deltas vs the baseline with spread; when the best
delta is within its own spread the verdict is "no_clear_winner" rather than a
coin-flip ranking.

What it measures (and the disclaimers a UI must carry):
- a FROZEN-backbone linear probe — relative signal, not fine-tune metrics. A
  fine-tuned backbone (especially the group's own prior checkpoint) tends to
  benefit MORE from extra pixels than fresh init features do;
- validation preprocessing only (no augmentation);
- nothing at all when ``backbone_init`` resolves to random weights — refused.

Cache relationship: each resolution's vectors land in the resolution-namespaced
EmbeddingCache era (``val_imagenet@<res>``) with ``prune=False``, so the probe
never deletes the production era, its candidate eras coexist, and the winning
resolution's era becomes the warm start for real training (the next real run
prunes the losers).
"""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image

from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.dataset import (
    DEFAULT_TRAIN_RESOLUTION,
    _list_split_images,
    _list_split_images_flat,
    find_nearest_bucket,
)
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.group_dataset import _list_class_images
from bittrainer.group_trainer import _embedding_preproc_sig
from bittrainer.group_validation import compute_multiclass_metrics
from bittrainer.model import backbone_feature_hash, create_model

logger = logging.getLogger(__name__)

DEFAULT_RESOLUTIONS = [384, 512, 640, 768]
_PROBE_EPOCHS = 120
_PROBE_LR = 1e-3
_PROBE_WEIGHT_DECAY = 1e-4


class ResolutionProbeError(RuntimeError):
    """The probe cannot produce a meaningful answer for this configuration."""


def _noop_progress(_msg: dict) -> None:
    pass


def _collect_group_sample(
    folder: Path, class_names: list[str], sample_size: int, rng: random.Random
) -> tuple[list[str], list[int], dict[str, int]]:
    per_class_quota = max(2, math.ceil(sample_size / max(len(class_names), 1)))
    paths: list[str] = []
    labels: list[int] = []
    counts: dict[str, int] = {}
    for class_idx, name in enumerate(class_names):
        class_paths = [
            str(p)
            for split in ("train", "val")
            for p in _list_class_images(folder, name, split)
        ]
        rng.shuffle(class_paths)
        chosen = class_paths[:per_class_quota]
        counts[name] = len(chosen)
        paths.extend(chosen)
        labels.extend([class_idx] * len(chosen))
    return paths, labels, counts


def _binary_split_images(folder: Path, kind: str) -> list[str]:
    out: list[str] = []
    for split in ("train", "val"):
        if kind == "positive":
            found = _list_split_images_flat(folder, split) or _list_split_images(
                folder, "positive", split
            )
        else:
            found = _list_split_images(folder, "negative", split)
        out.extend(str(p) for p in found)
    return out


def _collect_binary_sample(
    folder: Path, sample_size: int, rng: random.Random
) -> tuple[list[str], list[int], dict[str, int]]:
    positives = _binary_split_images(folder, "positive")
    negatives = _binary_split_images(folder, "negative")
    rng.shuffle(positives)
    rng.shuffle(negatives)
    # All positives up to half the budget, negatives matched 1:1 (the binary
    # trainer's own balance convention).
    pos = positives[: max(2, sample_size // 2)]
    neg = negatives[: max(2, len(pos))]
    counts = {"positive": len(pos), "negative": len(neg)}
    return neg + pos, [0] * len(neg) + [1] * len(pos), counts


def _read_dims(paths: list[str]) -> dict[str, tuple[int, int]]:
    dims: dict[str, tuple[int, int]] = {}
    for p in paths:
        try:
            with Image.open(p) as img:
                dims[p] = img.size
        except OSError:
            continue
    return dims


def _native_resolution_audit(
    dims: dict[str, tuple[int, int]], resolutions: list[int]
) -> dict:
    """Effective source resolution (sqrt of pixel area — comparable to the
    square-bucket side) so the UI can say WHY high-res gains are unlikely."""
    effective = sorted(math.sqrt(w * h) for w, h in dims.values())
    if not effective:
        return {"median_px": 0, "pct_below": {}}
    median = effective[len(effective) // 2]
    pct_below = {
        res: round(100.0 * sum(1 for e in effective if e < res) / len(effective), 1)
        for res in resolutions
    }
    return {"median_px": int(median), "pct_below": pct_below}


def _assign_folds(labels: list[int], folds: int, rng: random.Random) -> list[int]:
    """Stratified fold assignment, fixed once and shared by every resolution."""
    fold_of = [0] * len(labels)
    by_class: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)
    for indices in by_class.values():
        rng.shuffle(indices)
        for position, i in enumerate(indices):
            fold_of[i] = position % folds
    return fold_of


def _fit_fold(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    metric_name: str,
    seed: int,
    device: torch.device,
) -> float:
    """One linear probe fold on cached vectors; returns the fold metric."""
    torch.manual_seed(seed)  # identical head init per fold across resolutions
    head = nn.Linear(x_train.shape[1], num_classes).to(device)
    # Class-weighted CE so small classes in a small sample still register.
    counts = torch.bincount(y_train, minlength=num_classes).float().clamp(min=1)
    weight = (counts.sum() / counts).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight / weight.mean())
    optimizer = torch.optim.AdamW(head.parameters(), lr=_PROBE_LR, weight_decay=_PROBE_WEIGHT_DECAY)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    for _ in range(_PROBE_EPOCHS):  # full-batch: n is a few hundred
        optimizer.zero_grad()
        loss = loss_fn(head(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = head(x_val.to(device)).argmax(dim=1).cpu().tolist()
    metrics = compute_multiclass_metrics(y_val.tolist(), preds, num_classes)
    return float(metrics[metric_name])


def _precompute_crop_bboxes(
    request: dict, folder: Path, paths: list[str], cb: Callable[[dict], None]
) -> dict[str, list[int]]:
    crop_model = request.get("region_model_path") or request.get("face_model_path") or ""
    if not crop_model:
        return {}
    from bittrainer.face_crop import (
        FaceBBoxCache,
        precompute_region_bboxes,
        region_bbox_cache_name,
    )

    if request.get("region_model_path"):
        region_classes = list(request.get("region_classes") or [])
        selection = str(request.get("region_selection") or "highest_conf")
        cache_name = region_bbox_cache_name(crop_model, region_classes, selection)
        target_classes = region_classes or None
    else:
        cache_name = "face_bboxes.json"
        target_classes = None
        selection = "union"
    # Shares the training bbox cache, so detections done here are reused by the
    # real run (and vice versa).
    bbox_cache = FaceBBoxCache(folder / ".resize_cache" / cache_name)

    def _progress(done: int, total: int) -> None:
        cb({
            "type": "training_progress", "stage": "region_detection",
            "status_text": f"Resolution probe: detecting crop regions ({done}/{total})",
            "step": done, "total_steps": total,
        })

    precompute_region_bboxes(
        paths, bbox_cache, crop_model,
        target_classes=target_classes, selection=selection,
        device=request.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"),
        progress_fn=_progress,
    )
    return {p: bbox for p in paths if (bbox := bbox_cache.get(p))}


def run_resolution_probe(
    request: dict,
    progress_callback: Callable[[dict], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> dict:
    """Run the probe; see the module docstring for design and semantics.

    Request keys: ``mode`` ("group"|"binary"), ``folder``, ``class_names``
    (group mode), ``backbone_init``, ``model_size``, ``resolutions``,
    ``baseline_resolution`` (default 512, always included), ``sample_size``
    (default 300), ``folds`` (default 5), ``seed``, ``skin_normalise``,
    ``face_model_path`` / ``region_model_path`` / ``region_classes`` /
    ``region_selection``, ``embedding_cache_dir``, ``device``.
    """
    cb = progress_callback or _noop_progress
    stop = stop_check or (lambda: False)

    spec = request.get("backbone_init") or {}
    if str(spec.get("source") or "") == "random_init" or request.get("from_scratch"):
        raise ResolutionProbeError(
            "The resolution probe needs real backbone features; a random-init / "
            "from-scratch configuration has nothing meaningful to measure."
        )

    mode = str(request.get("mode") or "group")
    folder = Path(request["folder"])
    baseline = int(request.get("baseline_resolution") or DEFAULT_TRAIN_RESOLUTION)
    resolutions = sorted(
        {int(r) for r in (request.get("resolutions") or DEFAULT_RESOLUTIONS)} | {baseline}
    )
    sample_size = int(request.get("sample_size") or 300)
    folds = max(2, int(request.get("folds") or 5))
    seed = int(request.get("seed") or 0)
    rng = random.Random(seed)

    if mode == "group":
        class_names = list(request.get("class_names") or [])
        if len(class_names) < 2:
            raise ResolutionProbeError("Group resolution probe needs at least 2 classes.")
        paths, labels, per_class = _collect_group_sample(folder, class_names, sample_size, rng)
        num_classes = len(class_names)
        metric_name = "macro_f1"
    else:
        paths, labels, per_class = _collect_binary_sample(folder, sample_size, rng)
        num_classes = 2
        metric_name = "balanced_accuracy"

    dims = _read_dims(paths)
    kept = [(p, label) for p, label in zip(paths, labels) if p in dims]
    if len(kept) < folds * 2 or len({label for _p, label in kept}) < 2:
        raise ResolutionProbeError(
            f"Not enough readable labelled images to probe ({len(kept)} usable)."
        )
    paths = [p for p, _ in kept]
    labels = [label for _, label in kept]
    fold_of = _assign_folds(labels, folds, rng)
    audit = _native_resolution_audit(dims, resolutions)

    device = torch.device(
        request.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_size = str(request.get("model_size") or "nano")
    backbone = create_model(
        model_size=model_size, pretrained=wants_timm_pretrained(spec), num_classes=0
    )
    apply_backbone_init(backbone, spec)
    backbone = backbone.to(device).eval()
    backbone_hash = backbone_feature_hash(backbone)
    pooled_dim = int(backbone.num_features)
    cache_root = str(request.get("embedding_cache_dir") or (folder / ".embedding_cache"))

    face_bboxes = _precompute_crop_bboxes(request, folder, paths, cb)
    skin_normalise = bool(request.get("skin_normalise"))
    y = torch.tensor(labels, dtype=torch.long)

    results = []
    fold_scores_by_resolution: dict[int, list[float]] = {}
    for resolution in resolutions:
        if stop():
            raise ResolutionProbeError("Resolution probe cancelled.")
        started = time.monotonic()
        cb({
            "type": "training_progress", "stage": "embedding_build",
            "status_text": f"Resolution probe: embedding sample at {resolution}px",
            "resolution": resolution,
        })
        samples = [
            {
                "path": p,
                "bucket": find_nearest_bucket(*dims[p], resolution),
                "skin_normalise": skin_normalise,
                "face_bbox": face_bboxes.get(p),
            }
            for p in paths
        ]
        cache = EmbeddingCache(
            cache_root, backbone_hash, pooled_dim,
            preproc_sig=_embedding_preproc_sig(resolution),
        )
        embed_batch = max(8, int(64 * (DEFAULT_TRAIN_RESOLUTION / resolution) ** 2))
        stats = cache.ensure(
            samples, backbone, None,
            device=device, dtype=torch.float32, batch_size=embed_batch,
            stop_check=stop, prune=False,
        )
        vectors = []
        row_indices = []
        for i, sample in enumerate(samples):
            vector = cache.get_vector(sample["path"], None)
            if vector is not None:
                vectors.append(torch.from_numpy(vector).float())
                row_indices.append(i)
        if len(vectors) < folds * 2:
            raise ResolutionProbeError(
                f"Too few embeddings built at {resolution}px ({len(vectors)})."
            )
        x = torch.stack(vectors)
        y_res = y[row_indices]
        folds_of_rows = [fold_of[i] for i in row_indices]

        fold_scores: list[float] = []
        for fold in range(folds):
            train_mask = torch.tensor([f != fold for f in folds_of_rows])
            val_mask = ~train_mask
            if int(val_mask.sum()) == 0 or int(train_mask.sum()) == 0:
                continue
            fold_scores.append(
                _fit_fold(
                    x[train_mask], y_res[train_mask], x[val_mask], y_res[val_mask],
                    num_classes, metric_name, seed * 1000 + fold, device,
                )
            )
        mean = sum(fold_scores) / len(fold_scores)
        std = (
            math.sqrt(sum((s - mean) ** 2 for s in fold_scores) / max(len(fold_scores) - 1, 1))
            if len(fold_scores) > 1
            else 0.0
        )
        fold_scores_by_resolution[resolution] = fold_scores
        results.append({
            "resolution": resolution,
            "metric_mean": round(mean, 4),
            "metric_std": round(std, 4),
            "fold_scores": [round(s, 4) for s in fold_scores],
            "compute_multiplier": round((resolution / DEFAULT_TRAIN_RESOLUTION) ** 2, 2),
            "seconds": round(time.monotonic() - started, 1),
            "cache_built": stats["built"],
        })
        cb({
            "type": "training_progress", "stage": "training",
            "status_text": (
                f"Resolution probe: {resolution}px -> {metric_name} {mean:.3f} "
                f"(±{std:.3f}, {len(fold_scores)} folds)"
            ),
            "resolution": resolution,
        })

    # Paired deltas vs the baseline on identical folds.
    base_scores = fold_scores_by_resolution[baseline]
    for row in results:
        scores = fold_scores_by_resolution[row["resolution"]]
        paired = list(zip(scores, base_scores))
        deltas = [s - b for s, b in paired]
        delta_mean = sum(deltas) / len(deltas) if deltas else 0.0
        delta_std = (
            math.sqrt(sum((d - delta_mean) ** 2 for d in deltas) / max(len(deltas) - 1, 1))
            if len(deltas) > 1
            else 0.0
        )
        row["delta_mean"] = round(delta_mean, 4)
        row["delta_std"] = round(delta_std, 4)

    contenders = [r for r in results if r["resolution"] != baseline]
    best = max(contenders, key=lambda r: r["delta_mean"], default=None)
    if best is not None and best["delta_mean"] > max(best["delta_std"], 1e-4):
        verdict = {"kind": "winner", "resolution": best["resolution"]}
    else:
        verdict = {"kind": "no_clear_winner", "resolution": baseline}

    return {
        "metric_name": metric_name,
        "mode": mode,
        "sample_size": len(paths),
        "per_class_counts": per_class,
        "folds": folds,
        "seed": seed,
        "native_resolution": audit,
        "baseline_resolution": baseline,
        "results": results,
        "verdict": verdict,
        "backbone_hash": backbone_hash,
        "model_size": model_size,
        "crops_applied": bool(face_bboxes),
        "completed_at": None,  # stamped by the caller (no wall clock in-lib)
    }
