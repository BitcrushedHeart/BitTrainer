"""Start a group model from a BitVision trunk and its head (Bitcrush ISSUE-1721).

Every group model is retrained from a trunk that descends from the Apache
ConvNeXt V1 weights, never continued from its current model (whose trunk
started from timm's CC-BY-NC-4.0 ConvNeXt V2 weights).

A backbone candidate already holds a linear head for each single-label group
(``heads.groups.<group key>``), fed the post-norm pooled feature: the same
point a group model's ``head.fc`` reads (``model.pooled_features``). So the
group model can start exactly where the backbone's head is, answering what the
backbone answers. Two things differ: the backbone orders a group's classes as
sorted strings ("k10" before "k9"), the group model in Engine's class order,
so the rows are reordered; and a hidden-MLP head has no counterpart in the
backbone, so it keeps the trunk and starts its tail fresh, and says so.

The group trainer takes the result through ``backbone_init`` (a
``local_candidate`` whose ``checkpoint_path`` is the ``.safetensors`` written
here: every tensor whose key and shape match loads, the seeded ``head.fc``
included). It must not be placed as ``best.pt`` in the checkpoint folder: that
would make it the incumbent the new model is judged against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from bittrainer.lineage import extend, read_metadata, require_apache, to_json
from bittrainer.model import backbone_state_hash, create_model

logger = logging.getLogger(__name__)

_TAIL = ("head.fc.", "head.pre_logits.")


class GroupStartError(ValueError):
    """The trunk cannot start this group's model."""


@dataclass
class GroupStart:
    """A group model ready to train, and where it came from."""

    model: nn.Module
    class_names: list[str]
    size: str
    lineage: dict
    start_trunk_hash: str
    head_seeded: bool
    note: str
    path: Path | None = None


def _module_key(name: str) -> str:
    # As the backbone trainer names a head module (no '.' in a module key).
    return name.replace(".", "_") if name else name


def _head_classes(meta: dict, group_key: str, trunk: str) -> list[str]:
    try:
        groups = json.loads(meta.get("heads_groups_json") or "{}")
    except (TypeError, ValueError) as exc:
        raise GroupStartError(f"{trunk} has an unreadable head layout: {exc}") from exc
    classes = groups.get(_module_key(group_key)) if isinstance(groups, dict) else None
    if classes is None:
        raise GroupStartError(
            f"{trunk} has no head for group {group_key}: the backbone was not trained "
            "on it (a multi-label group trains yes/no heads, not a group head)"
        )
    return [str(name) for name in classes]


def _save(path: Path, model: nn.Module, fields: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(path), metadata=fields)
        return
    torch.save({"state_dict": state, **fields}, str(path))


def start_group_model(
    trunk_file,
    group_key: str,
    class_ids: list[str],
    size: str,
    *,
    out=None,
    head_hidden_size: int | None = None,
) -> GroupStart:
    """A group model for *group_key* (the pack's id, e.g. ``g6``) started from
    the backbone candidate *trunk_file*, its classes *class_ids* in Engine's
    order, at ConvNeXt V2 *size*.

    With a linear head (``head_hidden_size`` None) ``head.fc`` is the
    backbone's head for the group, its rows put in *class_ids* order, so the
    model answers as the backbone does. With a hidden-MLP head the trunk is
    kept and the head tail starts fresh.

    Refuses (:class:`~bittrainer.lineage.LineageError`) a trunk that does not
    descend from the Apache ConvNeXt V1 weights, and
    (:class:`GroupStartError`) a trunk of another size, one with no head for
    the group, or a head whose classes are not *class_ids*.

    *out* (optional) is where the start is written, with its ``lineage_json``:
    a ``.safetensors`` for the group trainer's ``backbone_init``, or a
    ``.pt`` holding ``state_dict`` and the same fields.
    """
    trunk_file = Path(trunk_file)
    name = f"the trunk {trunk_file.name}"
    meta = read_metadata(trunk_file)
    require_apache(meta, name)
    trunk_size = meta.get("convnextv2_size")
    if trunk_size != size:
        raise GroupStartError(
            f"{name} is a ConvNeXt V2 {trunk_size or 'of no recorded size'}, and the "
            f"group trains at {size}: use the BitVision trunk of that size"
        )
    classes = [str(class_id) for class_id in class_ids]
    if len(set(classes)) != len(classes):
        raise GroupStartError(f"the class ids repeat: {classes}")
    head_classes = _head_classes(meta, group_key, name)
    if set(head_classes) != set(classes):
        only_here = sorted(set(classes) - set(head_classes))
        only_head = sorted(set(head_classes) - set(classes))
        raise GroupStartError(
            f"the backbone's head for {group_key} has other classes than the group: "
            f"the group only has {only_here or 'none'}, the head only has "
            f"{only_head or 'none'}"
        )

    from safetensors.torch import load_file

    state = load_file(str(trunk_file))
    key = f"heads.groups.{_module_key(group_key)}"
    weight, bias = state.get(f"{key}.weight"), state.get(f"{key}.bias")
    if weight is None or bias is None:
        raise GroupStartError(f"{name} records a head for {group_key} but holds none")
    trunk_state = {k: v for k, v in state.items() if not k.startswith("heads.")}

    model = create_model(
        model_size=size,
        pretrained=False,
        num_classes=len(classes),
        head_hidden_size=head_hidden_size,
        drop_path_rate=0.0,
    )
    target = model.state_dict()
    wanted = {k for k in target if not k.startswith(_TAIL)}
    misfit = sorted(
        k
        for k in wanted
        if k not in trunk_state or trunk_state[k].shape != target[k].shape
    )
    if misfit:
        raise GroupStartError(
            f"{name} does not fit a ConvNeXt V2 {size} group model: {len(misfit)} "
            f"tensors missing or of another shape (first: {misfit[0]})"
        )
    seed = {k: trunk_state[k].to(target[k].dtype) for k in wanted}

    if head_hidden_size is None:
        order = [head_classes.index(class_id) for class_id in classes]
        seed["head.fc.weight"] = weight[order].to(target["head.fc.weight"].dtype)
        seed["head.fc.bias"] = bias[order].to(target["head.fc.bias"].dtype)
        seeded = True
        note = f"head.fc starts as the backbone's {group_key} head, in Engine's class order"
    else:
        seeded = False
        note = (
            f"a hidden-MLP head ({head_hidden_size} wide) has no counterpart in the "
            "backbone: the trunk is kept and the head tail starts fresh"
        )
        logger.warning("Group start for %s: %s", group_key, note)
    model.load_state_dict(seed, strict=False)

    trunk_hash = backbone_state_hash(trunk_state)
    record = extend(meta, {"kind": "group", "size": size, "from_trunk": trunk_hash})
    start = GroupStart(
        model=model,
        class_names=classes,
        size=size,
        lineage=record,
        start_trunk_hash=trunk_hash,
        head_seeded=seeded,
        note=note,
    )
    if out is not None:
        start.path = Path(out)
        fields = {
            "lineage_json": to_json(record),
            "start_trunk_hash": trunk_hash,
            "model_size": size,
            "group_key": group_key,
        }
        if start.path.suffix == ".safetensors":  # a header holds text only
            fields.update(
                convnextv2_size=size,
                class_names_json=json.dumps(classes),
                head_seeded="true" if seeded else "false",
            )
        else:
            fields.update(
                num_classes=len(classes), class_names=classes, head_seeded=seeded
            )
        _save(start.path, model, fields)
    return start
