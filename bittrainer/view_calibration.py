"""Two views, one checkpoint: per-view calibration for crop-assisted groups
(Bitcrush ISSUE-0993).

A crop-assisted model is served on the whole image AND on the crops its own
region head proposes, and whichever view is more sure of itself wins. For that
comparison to mean anything the two views have to be calibrated separately — a
crop is a different input distribution from a full frame — and the rule that
lets a crop overrule the full view has to be fitted, not assumed.

Finalisation therefore adds four keys to the promoted checkpoint:

``region_decode``
    ``score_threshold`` (tuned on the drawn validation boxes), ``max_regions``
    and ``crop_pad`` (how the training crops were cut, so serving cuts the same).
``view_calibration``
    ``full`` and ``crop``. Single-label: ``temperature`` + ``class_logit_bias``;
    multi-label: ``per_class_thresholds``. ``full`` repeats the checkpoint's
    top-level constants — those stay the full view, so Engine builds that know
    nothing of regions decode exactly as before. ``crop`` is fitted on the
    validation crops, or copied from ``full`` when there are too few of them.
    Ordinal cut-points are shared by both views.
``view_selection``
    ``crop_margin``: a crop wins when its confidence is at least the full
    view's plus this. ``crop_trigger``: a full view at least this confident is
    not worth proposing crops for. ``confidence`` / ``radius`` record HOW
    confidence was measured, so serving measures it the same way.
``view_metrics``
    What the views scored on validation, alone and combined, plus the
    proposer's precision / recall.

The pure functions here (:func:`view_confidence`, :func:`combine_exclusive`,
:func:`combine_multilabel`) are the reference the Engine's serving code mirrors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from bittrainer.regions import (
    DEFAULT_MAX_REGIONS,
    DEFAULT_SCORE_THRESHOLD,
    RegionProposal,
    decode_regions,
    forward_with_regions,
    has_region_head,
)

if TYPE_CHECKING:
    from bittrainer.group_dataset import GroupDataset
    from bittrainer.group_trainer import GroupTrainConfig
    from bittrainer.region_manifest import RegionManifest

logger = logging.getLogger(__name__)

# Below this many validation crops (or images that have crops) a fit is noise:
# the crop view borrows the full view's calibration and the selection rule
# keeps its defaults.
MIN_VIEW_VAL_CROPS = 30

DEFAULT_CROP_MARGIN = 0.0
DEFAULT_CROP_TRIGGER = 1.0

_MARGIN_GRID = (-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
_TRIGGER_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
# A cheaper trigger is taken when it costs no more than this much score.
_TRIGGER_TOLERANCE = 0.002
_SCORE_GRID = tuple(round(0.1 + 0.05 * i, 2) for i in range(17))  # 0.10 .. 0.90
_DETECTION_IOU = 0.5
# Floor used while collecting proposals to tune over (well under any threshold
# worth shipping).
_CANDIDATE_SCORE = 0.05

_DEFAULT_ORDINAL_RADIUS = 2


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def ordinal_radius(sigma: float | None, num_classes: int) -> int:
    """Neighbourhood radius for ordinal confidence, from the label softness.

    The same mapping as the Engine's ``effective_ordinal_radius`` (wider soft
    labels spread mass over more neighbours). The value used at tuning time is
    stored in ``view_selection`` so serving never has to re-derive it.
    """
    if sigma is None:
        radius = _DEFAULT_ORDINAL_RADIUS
    elif sigma <= 0.1:
        radius = 1
    elif sigma <= 0.5:
        radius = 2
    else:
        radius = 3
    return max(1, min(radius, max(1, num_classes)))


def view_confidence(
    probs,
    predicted: int,
    *,
    ordinal: bool = False,
    none_index: int = -1,
    radius: int = _DEFAULT_ORDINAL_RADIUS,
) -> float:
    """How sure one view is of its prediction.

    Plain groups: the predicted class's probability. Ordinal groups: the mass
    within ``radius`` classes of the prediction, ``__none__`` excluded — an
    ordinal head spreads probability over adjacent classes on purpose, so a
    single bin badly understates it.
    """
    row = [float(p) for p in probs]
    if predicted < 0 or predicted >= len(row):
        return 0.0
    if not ordinal:
        return row[predicted]
    lo = max(0, predicted - radius)
    hi = min(len(row), predicted + radius + 1)
    return sum(p for i, p in enumerate(row[lo:hi], start=lo) if i != none_index)


def _predict(probs, none_index: int, cut_points) -> int:
    """One view's class under the shipped decode (cut-points when fitted)."""
    row = np.asarray(probs, dtype=np.float64)
    if cut_points:
        from bittrainer.group_validation import ordinal_decode

        return int(ordinal_decode(row[None, :], none_index=none_index, cut_points=cut_points)[0])
    return int(row.argmax())


# ---------------------------------------------------------------------------
# Combining the views
# ---------------------------------------------------------------------------


class ViewChoice(NamedTuple):
    view: str  # "full" | "crop"
    crop_index: int | None
    predicted: int
    probs: list[float]
    confidence: float


def combine_exclusive(
    full_probs,
    crop_probs,
    *,
    margin: float = DEFAULT_CROP_MARGIN,
    trigger: float = DEFAULT_CROP_TRIGGER,
    none_index: int = -1,
    ordinal: bool = False,
    radius: int = _DEFAULT_ORDINAL_RADIUS,
    cut_points=None,
) -> ViewChoice:
    """Single-label: the full view, unless a crop is surer by ``margin``.

    Only the most confident crop competes. A crop that predicts ``__none__``
    never does: regions cannot be labelled ``__none__``, so a crop that looks
    like nothing is a bad crop, not evidence about the image. A full view
    already at ``trigger`` confidence is taken as is.
    """
    full = [float(p) for p in full_probs]
    full_pred = _predict(full, none_index, cut_points)
    full_conf = view_confidence(
        full, full_pred, ordinal=ordinal, none_index=none_index, radius=radius
    )
    choice = ViewChoice("full", None, full_pred, full, full_conf)
    if full_conf >= trigger:
        return choice
    best: ViewChoice | None = None
    for index, raw in enumerate(crop_probs):
        row = [float(p) for p in raw]
        pred = _predict(row, none_index, cut_points)
        if pred == none_index:
            continue
        conf = view_confidence(row, pred, ordinal=ordinal, none_index=none_index, radius=radius)
        if best is None or conf > best.confidence:
            best = ViewChoice("crop", index, pred, row, conf)
    if best is not None and best.confidence >= full_conf + margin:
        return best
    return choice


class MultiLabelChoice(NamedTuple):
    probs: list[float]
    predicted_classes: list[int]
    views: list[str]  # per class: which view supplied the probability


def combine_multilabel(
    full_probs,
    crop_probs,
    *,
    full_thresholds,
    crop_thresholds,
) -> MultiLabelChoice:
    """Multi-label: per class, the view that clears its own threshold by more.

    A class is predicted when either view puts it over that view's threshold;
    the reported probability comes from the view with the larger margin over
    its threshold (so a crop that finds a small object is not averaged away by
    a full frame that cannot see it).
    """
    full = [float(p) for p in full_probs]
    crops = [[float(p) for p in row] for row in crop_probs]
    probs: list[float] = []
    views: list[str] = []
    predicted: list[int] = []
    for c, p_full in enumerate(full):
        best_p, best_gap, best_view = p_full, p_full - float(full_thresholds[c]), "full"
        for row in crops:
            gap = row[c] - float(crop_thresholds[c])
            if gap > best_gap:
                best_p, best_gap, best_view = row[c], gap, "crop"
        probs.append(best_p)
        views.append(best_view)
        if best_gap >= 0.0:
            predicted.append(c)
    return MultiLabelChoice(probs=probs, predicted_classes=predicted, views=views)


# ---------------------------------------------------------------------------
# Crop-view calibration
# ---------------------------------------------------------------------------


def fit_crop_calibration(
    crop_logits: torch.Tensor | None,
    crop_labels: torch.Tensor | None,
    config: GroupTrainConfig,
    none_index: int,
    *,
    full: dict,
) -> dict:
    """Calibration constants for the crop view, from the validation crops.

    ``crop_logits`` must already carry the prior adjustment (the same one the
    full view gets), since the constants are fitted on what serving will feed
    them. Under :data:`MIN_VIEW_VAL_CROPS` crops the full view's constants are
    returned with ``fallback=True``.
    """
    n = 0 if crop_logits is None else int(crop_logits.shape[0])
    if config.multi_label:
        default = [float(t) for t in (full.get("per_class_thresholds") or [0.5] * config.num_classes)]
        if n < MIN_VIEW_VAL_CROPS:
            return {"per_class_thresholds": default, "fallback": True, "n_val": n}
        from bittrainer.group_validation import find_per_class_thresholds

        probs = torch.sigmoid(crop_logits.float()).numpy()
        labels = crop_labels.numpy().astype(np.int64)
        thresholds = find_per_class_thresholds(probs, labels)
        # A class with no positive crop keeps the full view's threshold rather
        # than the grid's 0.5 default.
        positives = labels.sum(axis=0)
        fitted = [
            float(thresholds[c]) if positives[c] > 0 else default[c]
            for c in range(config.num_classes)
        ]
        return {"per_class_thresholds": fitted, "fallback": False, "n_val": n}

    full_temperature = float(full.get("temperature") or 1.0)
    full_bias = [float(v) for v in (full.get("class_logit_bias") or [0.0] * config.num_classes)]
    if n < MIN_VIEW_VAL_CROPS:
        return {
            "temperature": full_temperature,
            "class_logit_bias": full_bias,
            "fallback": True,
            "n_val": n,
        }
    from bittrainer.generic.evaluation import _tune_softmax_calibration

    temperature, bias, _metrics = _tune_softmax_calibration(
        crop_logits, crop_labels, config, none_index
    )
    return {
        "temperature": float(temperature),
        "class_logit_bias": [float(v) for v in bias],
        "fallback": False,
        "n_val": n,
    }


# ---------------------------------------------------------------------------
# View selection
# ---------------------------------------------------------------------------


def _score_predictions(
    labels: list[int], preds: list[int], config: GroupTrainConfig, none_index: int
) -> tuple[float, dict]:
    from bittrainer.group_validation import compute_multiclass_metrics, compute_ordinal_metrics
    from bittrainer.selection import _metric_score

    metrics = compute_multiclass_metrics(labels, preds, config.num_classes)
    if config.ordinal:
        metrics.update(
            compute_ordinal_metrics(labels, preds, config.num_classes, none_index=none_index)
        )
    summary = {"macro_f1": float(metrics["macro_f1"]), "accuracy": float(metrics["micro_f1"])}
    if config.ordinal:
        summary["qwk"] = float(metrics.get("qwk") or 0.0)
    return float(_metric_score(metrics, config)), summary


def tune_view_selection(
    full_probs: np.ndarray,
    crop_probs: list[np.ndarray],
    labels,
    config: GroupTrainConfig,
    none_index: int,
    *,
    cut_points=None,
) -> tuple[dict, dict]:
    """Fit ``crop_margin`` / ``crop_trigger`` on validation images that have crops.

    ``full_probs[i]`` and ``crop_probs[i]`` (``[n_crops, C]``) are image ``i``'s
    calibrated views. Returns ``(view_selection, metrics)`` where ``metrics``
    holds the ``full`` and ``combined`` scores on those images. With fewer than
    :data:`MIN_VIEW_VAL_CROPS` images the defaults stand (``tuned=False``):
    the more confident view wins, and crops are always proposed.
    """
    label_list = [int(v) for v in labels]
    n = len(label_list)
    ordinal = bool(config.ordinal)
    radius = ordinal_radius(config.ordinal_sigma, config.num_classes) if ordinal else 0
    spec = dict(none_index=none_index, ordinal=ordinal, radius=radius, cut_points=cut_points)

    def _evaluate(margin: float, trigger: float) -> tuple[float, dict]:
        preds = [
            combine_exclusive(
                full_probs[i], crop_probs[i], margin=margin, trigger=trigger, **spec
            ).predicted
            for i in range(n)
        ]
        return _score_predictions(label_list, preds, config, none_index)

    selection = {
        "crop_margin": DEFAULT_CROP_MARGIN,
        "crop_trigger": DEFAULT_CROP_TRIGGER,
        "confidence": "ordinal_neighbourhood" if ordinal else "top1",
        "radius": radius,
        "tuned": False,
    }
    metrics: dict = {"n_val_images_with_crops": n}
    if n == 0:
        return selection, metrics
    # trigger 0.0 = the full view always stands: the no-crops baseline.
    metrics["full"] = _evaluate(DEFAULT_CROP_MARGIN, 0.0)[1]

    if n >= MIN_VIEW_VAL_CROPS:
        best_score, best_margin = None, DEFAULT_CROP_MARGIN
        for margin in _MARGIN_GRID:
            score, _ = _evaluate(margin, DEFAULT_CROP_TRIGGER)
            better = best_score is None or score > best_score + 1e-9
            # Among equals, the margin nearest zero: "the surer view wins".
            tie = (
                best_score is not None
                and abs(score - best_score) <= 1e-9
                and abs(margin) < abs(best_margin)
            )
            if better or tie:
                best_score, best_margin = score, margin
        best_trigger = DEFAULT_CROP_TRIGGER
        for trigger in _TRIGGER_GRID:
            score, _ = _evaluate(best_margin, trigger)
            if score >= best_score - _TRIGGER_TOLERANCE:
                best_trigger = trigger
                break
        selection.update(crop_margin=best_margin, crop_trigger=best_trigger, tuned=True)

    metrics["combined"] = _evaluate(selection["crop_margin"], selection["crop_trigger"])[1]
    return selection, metrics


# ---------------------------------------------------------------------------
# Proposer threshold
# ---------------------------------------------------------------------------


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match_regions(
    proposals: list[RegionProposal],
    boxes: list[tuple[float, float, float, float, int]],
    *,
    iou: float = _DETECTION_IOU,
) -> tuple[int, int, int]:
    """``(true positives, false positives, false negatives)`` for one image.

    Greedy, highest score first; a drawn box is matched at most once, and only
    by a proposal of its own region type.
    """
    taken = [False] * len(boxes)
    tp = fp = 0
    for proposal in sorted(proposals, key=lambda p: p.score, reverse=True):
        best_index, best_iou = -1, iou
        for index, box in enumerate(boxes):
            if taken[index] or int(box[4]) != proposal.type:
                continue
            overlap = box_iou(proposal.box, box[:4])
            if overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0:
            taken[best_index] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, taken.count(False)


def tune_score_threshold(images: list[dict]) -> tuple[float, dict]:
    """The proposal score threshold with the best F1 on the validation boxes.

    Each image is ``{"proposals": [...], "boxes": [(x1, y1, x2, y2, type)],
    "supervised": {type, ...}}``. Proposals of a type the user did not review on
    that image are unknown and left out of the count. Without any validation
    box the default threshold stands (``tuned=False``).
    """
    n_boxes = sum(len(i["boxes"]) for i in images)
    default = {
        "tuned": False,
        "f1": None,
        "precision": None,
        "recall": None,
        "n_val_boxes": n_boxes,
        "n_val_images": len(images),
    }
    if n_boxes == 0:
        return DEFAULT_SCORE_THRESHOLD, default

    best: tuple[float, float, float, float] | None = None  # f1, threshold, precision, recall
    for threshold in _SCORE_GRID:
        tp = fp = fn = 0
        for image in images:
            kept = [
                p
                for p in image["proposals"]
                if p.score >= threshold and p.type in image["supervised"]
            ]
            a, b, c = match_regions(kept, list(image["boxes"]))
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        better = best is None or f1 > best[0] + 1e-9
        # Among equals, the threshold nearest the default.
        tie = (
            best is not None
            and abs(f1 - best[0]) <= 1e-9
            and abs(threshold - DEFAULT_SCORE_THRESHOLD) < abs(best[1] - DEFAULT_SCORE_THRESHOLD)
        )
        if better or tie:
            best = (f1, threshold, precision, recall)
    assert best is not None
    return best[1], {**default, "tuned": True, "f1": best[0], "precision": best[2], "recall": best[3]}


# ---------------------------------------------------------------------------
# The finaliser
# ---------------------------------------------------------------------------


def _fixed_loader(dataset, batch_size: int, collate_fn) -> tuple[DataLoader, list[int]]:
    """An in-process loader over a pinned batch order, and that order flattened
    — the only way a row of logits can be traced back to its sample."""
    from bittrainer.group_dataset import build_group_bucket_sampler
    from bittrainer.training_state import _FixedBatchSampler

    schedule = [list(b) for b in build_group_bucket_sampler(dataset, batch_size=batch_size)]
    loader = DataLoader(
        dataset, batch_sampler=_FixedBatchSampler(schedule), collate_fn=collate_fn, num_workers=0
    )
    return loader, [i for batch in schedule for i in batch]


def _prior_delta(ckpt: dict, num_classes: int) -> torch.Tensor | None:
    natural = ckpt.get("class_log_prior_natural")
    effective = ckpt.get("class_log_prior_train_effective")
    if not isinstance(natural, dict) or not isinstance(effective, dict):
        return None
    from bittrainer.priors import _prior_logit_delta

    try:
        tau = float(ckpt.get("prior_tau", 1.0))
    except (TypeError, ValueError):
        tau = 1.0
    return torch.tensor(_prior_logit_delta(natural, effective, num_classes, tau), dtype=torch.float32)


def _calibrated_probs(logits: torch.Tensor, calibration: dict, prior: torch.Tensor | None) -> np.ndarray:
    """Single-label view probabilities under the shipped decode order:
    prior adjustment -> temperature -> class bias -> softmax."""
    adjusted = logits.float() if prior is None else logits.float() + prior
    scaled = adjusted / max(float(calibration.get("temperature") or 1.0), 1e-6)
    bias = calibration.get("class_logit_bias")
    if bias and len(bias) == scaled.shape[-1]:
        scaled = scaled + torch.tensor([float(v) for v in bias], dtype=torch.float32)
    return torch.softmax(scaled, dim=-1).numpy()


@torch.no_grad()
def _collect_region_pass(model, loader, config: GroupTrainConfig, device, dtype):
    """Full-view logits and decoded proposals for every region validation image."""
    from bittrainer.gpu_augment import apply_val_transform

    model.eval()
    memory_format = torch.channels_last if config.channels_last else None
    logits_all: list[torch.Tensor] = []
    rows: list[dict] = []
    for images, _labels, targets in loader:
        images = apply_val_transform(
            images.to(device, non_blocking=True), dtype=dtype, memory_format=memory_format
        )
        with torch.amp.autocast(
            device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)
        ):
            logits, out = forward_with_regions(model, images)
        logits_all.append(logits.float().cpu())
        proposals = decode_regions(
            out, score_threshold=_CANDIDATE_SCORE, max_regions=DEFAULT_MAX_REGIONS
        )
        boxes = targets.boxes.tolist()
        owners = targets.batch_index.tolist()
        kinds = targets.types.tolist()
        for b in range(images.shape[0]):
            rows.append(
                {
                    "proposals": proposals[b],
                    "boxes": [
                        (*boxes[i], kinds[i]) for i in range(len(boxes)) if owners[i] == b
                    ],
                    "supervised": {
                        k for k, flag in enumerate(targets.supervised[b].tolist()) if flag
                    },
                }
            )
    if not logits_all:
        return torch.zeros(0, config.num_classes), rows
    return torch.cat(logits_all, dim=0), rows


def finalize_region_views(
    config: GroupTrainConfig,
    *,
    checkpoint_path: str | None,
    manifest: RegionManifest,
    val_ds: GroupDataset,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 32,
    cb: Callable[[dict], None] | None = None,
) -> dict:
    """Calibrate the crop view, fit the selection rule and tune the proposer on
    the promoted checkpoint, then persist the four region keys into it.

    Runs after promotion on whichever checkpoint won. A winner without a region
    head (an incumbent from before the group had regions, or a head-only
    candidate that had to drop its head) is left untouched and reported as
    ``{"status": "no_region_head"}``. Never raises: a failure here must not
    undo a finished training run.
    """
    try:
        return _finalize_region_views(
            config,
            checkpoint_path=checkpoint_path,
            manifest=manifest,
            val_ds=val_ds,
            device=device,
            dtype=dtype,
            batch_size=max(1, int(batch_size or 32)),
            cb=cb,
        )
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning("Region view finalisation failed", exc_info=True)
        return {"status": "error", "error": str(exc)}


def _finalize_region_views(
    config: GroupTrainConfig,
    *,
    checkpoint_path: str | None,
    manifest: RegionManifest,
    val_ds: GroupDataset,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    cb: Callable[[dict], None] | None,
) -> dict:
    from bittrainer.collate import _collate_bucket_batch, _collate_multilabel_batch
    from bittrainer.generic.evaluation import _collect_val_logits
    from bittrainer.group_dataset import GroupDataset
    from bittrainer.model import load_checkpoint
    from bittrainer.region_manifest import norm_path
    from bittrainer.selection import _resolve_none_index

    if not checkpoint_path:
        return {"status": "no_checkpoint"}
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "region_head" not in ckpt:
        return {"status": "no_region_head"}
    model = load_checkpoint(
        checkpoint_path,
        device=str(device),
        dtype=dtype,
        model_size=config.backbone_variant,
        num_classes=config.num_classes,
    ).to(device)
    if not has_region_head(model):
        return {"status": "no_region_head"}
    if cb is not None:
        cb(
            {
                "type": "training_progress",
                "stage": "calibrating",
                "status_text": "Calibrating the crop view and the region proposer",
            }
        )

    none_index = _resolve_none_index(config.class_names)
    collate = _collate_multilabel_batch if config.multi_label else _collate_bucket_batch
    dataset_kwargs = dict(
        split="val",
        multi_label=config.multi_label,
        skin_normalise=config.skin_normalise,
        group_name=config.group_name or "",
        train_resolution=config.train_resolution,
        scan_disk=False,
    )
    group_folder = config.group_folder

    # --- Full view + proposals on every drawn validation image ---
    region_ds = GroupDataset(
        group_folder, config.class_names, region_manifest=manifest, **dataset_kwargs
    )
    if len(region_ds):
        loader, order = _fixed_loader(region_ds, batch_size, collate)
        full_logits, region_rows = _collect_region_pass(model, loader, config, device, dtype)
        region_paths = [region_ds.samples[i]["path"] for i in order]
    else:
        full_logits, region_rows, region_paths = torch.zeros(0, config.num_classes), [], []
    score_threshold, detector_metrics = tune_score_threshold(region_rows)

    # --- Crop view on the validation crops ---
    crop_paths = manifest.crop_paths_by_class("val")
    crop_logits = crop_labels = None
    crop_sources: list[str] = []
    if any(crop_paths.values()):
        crop_ds = GroupDataset(
            group_folder, config.class_names, extra_paths=crop_paths, **dataset_kwargs
        )
        if len(crop_ds):
            loader, order = _fixed_loader(crop_ds, batch_size, collate)
            crop_logits, crop_labels = _collect_val_logits(model, loader, config, device, dtype)
            for i in order:
                crop = manifest.crop_for(crop_ds.samples[i]["path"])
                crop_sources.append(crop.source if crop is not None else "")
    del model

    # --- Calibration per view ---
    if config.multi_label:
        full_calibration: dict = {
            "per_class_thresholds": [
                float(t)
                for t in (ckpt.get("per_class_thresholds") or [0.5] * config.num_classes)
            ]
        }
        prior = None
        adjusted_crop_logits = crop_logits
    else:
        full_calibration = {
            "temperature": float(ckpt.get("temperature") or 1.0),
            "class_logit_bias": [
                float(v) for v in (ckpt.get("class_logit_bias") or [0.0] * config.num_classes)
            ],
        }
        prior = _prior_delta(ckpt, config.num_classes)
        adjusted_crop_logits = (
            crop_logits
            if crop_logits is None or prior is None
            else crop_logits.float() + prior
        )
    crop_calibration = fit_crop_calibration(
        adjusted_crop_logits, crop_labels, config, none_index, full=full_calibration
    )

    # --- Selection rule, on labelled validation images that have crops ---
    n_val_crops = 0 if crop_logits is None else int(crop_logits.shape[0])
    label_by_path = {norm_path(s["path"]): s["label"] for s in val_ds.samples}
    crops_by_source: dict[str, list[int]] = {}
    for row, source in enumerate(crop_sources):
        crops_by_source.setdefault(norm_path(source), []).append(row)

    paired: list[tuple[int, list[int], object]] = []  # (region row, crop rows, label)
    for row, path in enumerate(region_paths):
        image = manifest.image_for(path)
        if image is None:
            continue
        crop_rows = crops_by_source.get(norm_path(image.source), [])
        label = next(
            (label_by_path[norm_path(p)] for p in image.paths if norm_path(p) in label_by_path),
            None,
        )
        if crop_rows and label is not None:
            paired.append((row, crop_rows, label))

    view_metrics: dict = {
        "n_val_crops": n_val_crops,
        "n_val_images_with_crops": len(paired),
        "detector": detector_metrics,
    }
    if config.multi_label:
        selection = {
            "crop_margin": DEFAULT_CROP_MARGIN,
            "crop_trigger": DEFAULT_CROP_TRIGGER,
            "confidence": "threshold_margin",
            "radius": 0,
            "tuned": False,
        }
        if paired:
            view_metrics.update(
                _multilabel_view_metrics(
                    paired, full_logits, crop_logits, full_calibration, crop_calibration, config
                )
            )
    else:
        cut_points = ckpt.get("ordinal_cut_points") if config.ordinal else None
        if paired:
            full_probs = _calibrated_probs(full_logits, full_calibration, prior)
            crop_probs = _calibrated_probs(crop_logits, crop_calibration, prior)
            selection, selection_metrics = tune_view_selection(
                np.stack([full_probs[row] for row, _, _ in paired]),
                [crop_probs[rows] for _, rows, _ in paired],
                [int(label) for _, _, label in paired],
                config,
                none_index,
                cut_points=cut_points,
            )
            view_metrics.update(
                {k: v for k, v in selection_metrics.items() if k in ("full", "combined")}
            )
        else:
            selection, _ = tune_view_selection(
                np.zeros((0, config.num_classes)), [], [], config, none_index
            )
        if n_val_crops:
            crop_probs_all = _calibrated_probs(crop_logits, crop_calibration, prior)
            preds = [_predict(row, none_index, cut_points) for row in crop_probs_all]
            view_metrics["crop"] = _score_predictions(
                [int(v) for v in crop_labels.tolist()], preds, config, none_index
            )[1]

    payload = {
        "region_decode": {
            "score_threshold": float(score_threshold),
            "max_regions": int(DEFAULT_MAX_REGIONS),
            "crop_pad": float(manifest.crop_pad),
        },
        "view_calibration": {"full": full_calibration, "crop": crop_calibration},
        "view_selection": selection,
        "view_metrics": view_metrics,
    }
    # Re-read before writing: the dict above was loaded before the model was.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt.update(payload)
    torch.save(ckpt, checkpoint_path)
    return {"status": "ok", **payload}


def _multilabel_view_metrics(
    paired, full_logits, crop_logits, full_calibration: dict, crop_calibration: dict, config
) -> dict:
    """Full-view vs combined multi-label scores on images that have crops."""
    from bittrainer.group_validation import compute_multilabel_metrics

    full_probs = torch.sigmoid(full_logits.float()).numpy()
    crop_probs = torch.sigmoid(crop_logits.float()).numpy()
    full_t = np.asarray(full_calibration["per_class_thresholds"], dtype=np.float64)
    crop_t = np.asarray(crop_calibration["per_class_thresholds"], dtype=np.float64)
    labels = np.stack(
        [torch.as_tensor(label).numpy().astype(np.int64) for _, _, label in paired]
    )
    full_pred = np.stack(
        [(full_probs[row] >= full_t).astype(np.int64) for row, _, _ in paired]
    )
    combined_pred = np.zeros_like(full_pred)
    for i, (row, crop_rows, _label) in enumerate(paired):
        choice = combine_multilabel(
            full_probs[row], crop_probs[crop_rows], full_thresholds=full_t, crop_thresholds=crop_t
        )
        combined_pred[i, choice.predicted_classes] = 1

    def _summary(pred: np.ndarray) -> dict:
        metrics = compute_multilabel_metrics(labels, pred, config.num_classes)
        return {"macro_f1": float(metrics["macro_f1"])}

    return {"full": _summary(full_pred), "combined": _summary(combined_pred)}
