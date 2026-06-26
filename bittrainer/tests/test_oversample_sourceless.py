from __future__ import annotations

from bittrainer.group_dataset import GroupDataset, rare_group_none_target


class _FakeCache:
    """Minimal SmartCache stand-in: only iter_sourceless is exercised by the
    sourceless GroupDataset path."""

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def iter_sourceless(self) -> list[dict]:
        return list(self._samples)


def _base_samples() -> list[dict]:
    # 2 each of class 0, class 1, __none__ (idx 2), already equalised.
    out: list[dict] = []
    for label in (0, 0, 1, 1, 2, 2):
        out.append({"label": label, "split": "train", "path": f"p{len(out)}", "bucket": (256, 256)})
    return out


def _counts(ds: GroupDataset) -> dict[int, int]:
    c: dict[int, int] = {}
    for s in ds.samples:
        c[s["label"]] = c.get(s["label"], 0) + 1
    return c


def test_rare_group_none_target_formula():
    # ceil(1.5 * (max_count=2 * non_none_classes=2)) = 6.
    assert rare_group_none_target(2, 2) == 6
    # Never reduces below max_count.
    assert rare_group_none_target(10, 0) == 10


def test_sourceless_oversample_off_by_default():
    ds = GroupDataset(
        "/tmp/g", ["a", "b", "__none__"], split="train",
        cache=_FakeCache(_base_samples()), sourceless=True,
    )
    assert _counts(ds) == {0: 2, 1: 2, 2: 2}


def test_sourceless_set_oversample_none_reaches_target():
    ds = GroupDataset(
        "/tmp/g", ["a", "b", "__none__"], split="train",
        cache=_FakeCache(_base_samples()), sourceless=True,
    )
    ds.set_oversample_none(True)
    assert ds.oversample_none is True
    assert _counts(ds) == {0: 2, 1: 2, 2: 6}

    # Toggling back restores the un-oversampled base (idempotent re-derive).
    ds.set_oversample_none(False)
    assert _counts(ds) == {0: 2, 1: 2, 2: 2}


def test_sourceless_oversample_at_construction():
    ds = GroupDataset(
        "/tmp/g", ["a", "b", "__none__"], split="train",
        cache=_FakeCache(_base_samples()), sourceless=True,
        oversample_none=True,
    )
    assert _counts(ds) == {0: 2, 1: 2, 2: 6}


def test_sourceless_val_split_not_oversampled():
    val = [{"label": 2, "split": "val", "path": "v0", "bucket": (256, 256)}]
    ds = GroupDataset(
        "/tmp/g", ["a", "b", "__none__"], split="val",
        cache=_FakeCache(val), sourceless=True, oversample_none=True,
    )
    assert _counts(ds) == {2: 1}
