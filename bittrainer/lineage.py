"""Where a model's weights came from (Bitcrush ISSUE-1721).

The owner's rule: every backbone, distilled size and group model descends from
the Apache-2.0 ConvNeXt V1 conversion (``bittrainer.convnext_v1_init``), never
from timm's CC-BY-NC-4.0 ConvNeXt V2 weights. Each of them records how, as
``lineage_json`` in its metadata::

    {"root": "apache-convnext-v1:<timm weight name>",
     "steps": [{"kind": "backbone" | "distil" | "group",
                "size": "<convnextv2 size>",
                "from_trunk": "<backbone_state_hash> or null"}, ...]}

one step per stage it went through. A step may carry more (a distilled
student names its own ``init`` root). Engine reads it through this module.

Files written before ``lineage_json`` existed are read from the fields they
do have. A candidate whose run converted the V1 weights itself (``--init
apache`` at BitTrainer 5982d6e: ``initialization_source`` is
``convnext_v1_converted``, the weights and their licence are named) has an
Apache root and one ``backbone`` step; a bare converted trunk
(``converted_from`` + ``license``) is a root with no steps. Nothing else is
guessed: a recorded lineage is believed over those fields, and an older
distilled student is refused, since its teacher's start is not recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT_PREFIX = "apache-convnext-v1:"
STEP_KINDS = frozenset({"backbone", "distil", "group"})

# The licences convnext_v1_init accepts (kept here so reading a lineage never
# imports timm).
_PERMISSIVE = frozenset({"apache-2.0", "mit"})
_CONVERTED = "convnext_v1_converted"


class LineageError(ValueError):
    """A model does not descend from the Apache ConvNeXt V1 weights."""


def _is_v1_name(name: str) -> bool:
    # V1 weights are ``convnext_<size>.<tag>``; V2 ones ``convnextv2_...``.
    return name.startswith("convnext_") and "." in name


def _root_for(model_id: object) -> str | None:
    name = str(model_id or "").strip()
    if name.startswith("timm/"):
        name = name[len("timm/") :]
    return ROOT_PREFIX + name if _is_v1_name(name) else None


def _parsed(value: object) -> dict | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _valid_step(step: object) -> bool:
    return (
        isinstance(step, dict)
        and step.get("kind") in STEP_KINDS
        and isinstance(step.get("size"), str)
        and (step.get("from_trunk") is None or isinstance(step.get("from_trunk"), str))
    )


def _recorded(meta: dict) -> tuple[dict | None, str | None]:
    """A recorded lineage (``lineage_json``, or *meta* itself when it is one)
    and, when it is not an Apache one, why."""
    record = meta if "root" in meta and "steps" in meta else meta.get("lineage_json")
    record = _parsed(record)
    if record is None:
        return None, "its lineage_json cannot be read"
    root = record.get("root")
    steps = record.get("steps")
    if not isinstance(root, str) or not root.startswith(ROOT_PREFIX):
        return (
            None,
            f"its lineage starts at {root!r}, not the Apache ConvNeXt V1 weights",
        )
    if not _is_v1_name(root[len(ROOT_PREFIX) :]):
        return None, f"its lineage root {root!r} names no ConvNeXt V1 weights"
    if not isinstance(steps, list) or not all(_valid_step(s) for s in steps):
        return None, "its lineage_json has unreadable steps"
    return {"root": root, "steps": [dict(s) for s in steps]}, None


def _derived(meta: dict) -> tuple[dict | None, str]:
    """The lineage of a file written before ``lineage_json``, or why none."""
    source = meta.get("initialization_source")
    if source == _CONVERTED:
        licence = str(meta.get("pretrained_license") or "").lower()
        root = _root_for(meta.get("pretrained_model_id"))
        if root is None:
            return None, (
                f"its start {meta.get('pretrained_model_id')!r} is not a ConvNeXt V1 "
                "weight"
            )
        if licence not in _PERMISSIVE or _truthy(meta.get("non_commercial_only")):
            return (
                None,
                f"its start {root} is declared {licence or 'without a licence'}",
            )
        if meta.get("distilled_from"):
            return None, (
                "it was distilled from a teacher whose lineage it does not record"
            )
        step = {
            "kind": "backbone",
            "size": str(meta.get("convnextv2_size") or ""),
            "from_trunk": None,
        }
        return {"root": root, "steps": [step]}, ""
    if meta.get("converted_from"):
        licence = str(meta.get("license") or "").lower()
        root = _root_for(meta.get("converted_from"))
        if root is not None and licence in _PERMISSIVE:
            return {"root": root, "steps": []}, ""
        return None, (
            f"it was converted from {meta.get('converted_from')!r} "
            f"({licence or 'no licence'}), not Apache ConvNeXt V1 weights"
        )
    if source == "random_init":
        return (
            None,
            "it was started from random weights, not the Apache ConvNeXt V1 ones",
        )
    if source in ("timm_pretrained", "temporary_timm_pretrained_fallback"):
        model = meta.get("pretrained_model_id") or "timm's ConvNeXt V2 weights"
        licence = meta.get("pretrained_license") or "CC-BY-NC-4.0"
        return None, f"it was started from {model} ({licence})"
    return None, "it records no lineage and no Apache ConvNeXt V1 start"


def _resolve(meta: dict | None) -> tuple[dict | None, str]:
    if not isinstance(meta, dict) or not meta:
        return None, "it records no lineage and no Apache ConvNeXt V1 start"
    if ("root" in meta and "steps" in meta) or meta.get("lineage_json") is not None:
        record, why = _recorded(meta)
        return record, why or ""
    return _derived(meta)


def lineage_of(meta: dict | None) -> dict | None:
    """The full lineage ``{"root", "steps"}`` of a model from its metadata (a
    candidate header, a checkpoint's fields, or a lineage itself), or None when
    it does not descend from the Apache ConvNeXt V1 weights."""
    return _resolve(meta)[0]


def apache_root(meta: dict | None) -> str | None:
    """``apache-convnext-v1:<name>`` when *meta* descends from the Apache
    ConvNeXt V1 weights, else None."""
    record = lineage_of(meta)
    return record["root"] if record else None


def require_apache(meta: dict | None, what: str) -> str:
    """The Apache root of *meta*; raises :class:`LineageError` naming *what*
    and the plain reason when there is none."""
    record, why = _resolve(meta)
    if record is None:
        raise LineageError(
            f"{what} does not descend from the Apache ConvNeXt V1 weights: {why}. "
            "Every model must start from them (Bitcrush ISSUE-1721)."
        )
    return record["root"]


def extend(meta: dict | None, step: dict) -> dict:
    """The lineage of a model made from *meta* by one more *step*.

    Raises :class:`LineageError` when *meta* has no Apache root, and
    ``ValueError`` for a step that is not one of :data:`STEP_KINDS` with a
    size and a ``from_trunk`` (a hash or None).
    """
    if not _valid_step(step):
        raise ValueError(
            f"a lineage step needs kind in {sorted(STEP_KINDS)}, a size and a "
            f"from_trunk (a hash or None), got {step!r}"
        )
    require_apache(meta, "the model this starts from")
    record = lineage_of(meta)
    return {"root": record["root"], "steps": [*record["steps"], dict(step)]}


def to_json(record: dict) -> str:
    """*record* as the text a ``lineage_json`` field holds."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def read_metadata(path) -> dict:
    """The metadata of a model file: a safetensors header, or the fields
    (everything but tensors) of a torch checkpoint dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return dict(handle.metadata() or {})
    import torch

    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, value in data.items()
        if key != "state_dict" and not torch.is_tensor(value)
    }


def read_lineage(path) -> dict | None:
    """:func:`lineage_of` the model file at *path*."""
    return lineage_of(read_metadata(path))


def trunk_hash(path) -> str:
    """``backbone_state_hash`` of the trunk in a model file: a backbone
    candidate (its ``heads.*`` left out) or a group checkpoint's state dict.
    The same on the VM and at home, whatever the file is called."""
    from bittrainer.model import backbone_state_hash

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        import torch

        data = torch.load(str(path), map_location="cpu", weights_only=True)
        state = data.get("state_dict", data) if isinstance(data, dict) else {}
    return backbone_state_hash(
        {key: value for key, value in state.items() if not key.startswith("heads.")}
    )
