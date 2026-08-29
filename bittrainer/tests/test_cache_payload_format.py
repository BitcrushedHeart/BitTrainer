"""SmartCache ``.pt`` payload format — tensors, not proto-2 pickled ndarrays.

Bitcrush ISSUE-0847 cache audit: ``torch.save`` of a numpy uint8 array goes
through pickle protocol 2, which has no BINBYTES, so every byte >= 0x80 is
re-encoded as two UTF-8 bytes (~1.44x inflation) and ``torch.load`` is ~5x
slower. Storing a ``torch.Tensor`` writes the raw storage instead.

Both formats must coexist (no CACHE_VERSION bump) and an in-place converter
must upgrade legacy files atomically.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from bittrainer.smart_cache import CACHE_VERSION, SmartCache, payload_tensor
from bittrainer.tools import convert_cache_payloads as conv

BUCKET = (64, 48)  # (w, h)


def _rng_array(seed: int = 0, shape=(3, 48, 64)) -> np.ndarray:
    # Full 0..255 range so plenty of bytes >= 0x80 exercise the inflation.
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _make_source(tmp_path: Path, name: str, seed: int) -> str:
    src = tmp_path / "src" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(np.random.default_rng(seed).bytes(512))
    return os.path.normpath(str(src))


def _sample(path: str) -> dict:
    return {"path": path, "bucket": BUCKET, "label": 1, "split": "train", "concept_name": "c"}


def _build_cache(
    tmp_path: Path, n: int = 1
) -> tuple[SmartCache, list[dict], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    samples = []
    for i in range(n):
        p = _make_source(tmp_path, f"img{i}.bin", seed=100 + i)
        arrays[p] = _rng_array(seed=i)
        samples.append(_sample(p))
    cache = SmartCache(tmp_path / "cache", tqdm_enabled=False)
    cache.prepare(samples, lambda s: arrays[s["path"]], num_workers=1)
    return cache, samples, arrays


def _pt_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*.pt"))


def _write_legacy(pt_path: Path, arr: np.ndarray, **meta) -> None:
    """Hand-write a payload exactly as the pre-fix writer did (ndarray)."""
    payload = {
        "tensor": arr,
        "__cache_version": CACHE_VERSION,
        "__modeltype": "convnext_v2",
        "__resolution": f"{BUCKET[0]}x{BUCKET[1]}",
        "__bucket": tuple(BUCKET),
        "__source_path": "x",
        "__source_mtime": 0.0,
        "__source_hash": "0" * 16,
        "__skin_normalise": False,
        "__face_bbox": None,
        "__face_model_sig": "none",
    }
    payload.update(meta)
    torch.save(payload, pt_path)


def _non_pt_leftovers(cache_dir: Path) -> list[Path]:
    keep = {"cache.json", "cache.json.bak"}
    return [p for p in cache_dir.iterdir() if p.suffix != ".pt" and p.name not in keep]


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


def test_writer_stores_torch_tensor_and_roundtrips_bit_identically(tmp_path):
    cache, samples, arrays = _build_cache(tmp_path)
    (pt,) = _pt_files(tmp_path / "cache")
    raw = torch.load(pt, weights_only=False, map_location="cpu")
    assert isinstance(raw["tensor"], torch.Tensor), "writer must save a torch.Tensor payload"
    assert raw["tensor"].dtype == torch.uint8 and raw["tensor"].is_contiguous()
    for key in (
        "__cache_version",
        "__modeltype",
        "__resolution",
        "__bucket",
        "__source_path",
        "__source_mtime",
        "__source_hash",
        "__skin_normalise",
        "__face_bbox",
        "__face_model_sig",
    ):
        assert key in raw, key

    got, meta = cache.get(samples[0]["path"])
    assert isinstance(got, torch.Tensor) and got.dtype == torch.uint8
    assert np.array_equal(got.numpy(), arrays[samples[0]["path"]])
    assert "tensor" not in meta and meta["__bucket"] == tuple(BUCKET)


def test_legacy_ndarray_payload_still_reads_identically(tmp_path):
    cache, samples, arrays = _build_cache(tmp_path)
    (pt,) = _pt_files(tmp_path / "cache")
    arr = arrays[samples[0]["path"]]
    _write_legacy(pt, arr, __source_path=samples[0]["path"])
    raw = torch.load(pt, weights_only=False, map_location="cpu")
    assert isinstance(raw["tensor"], np.ndarray)  # precondition: legacy on disk

    got, meta = cache.get(samples[0]["path"])
    assert isinstance(got, torch.Tensor) and got.dtype == torch.uint8
    assert np.array_equal(got.numpy(), arr)
    # Sourceless metadata path reads the same file.
    (entry,) = cache.iter_sourceless()
    assert entry["bucket"] == tuple(BUCKET) and entry["_sourceless"] is True


def test_payload_tensor_accepts_both_formats():
    arr = _rng_array(seed=7)
    assert isinstance(payload_tensor({"tensor": arr}), torch.Tensor)
    assert np.array_equal(payload_tensor({"tensor": arr}).numpy(), arr)
    t = torch.from_numpy(arr.copy())
    assert payload_tensor({"tensor": t}) is t
    assert payload_tensor({}) is None
    assert payload_tensor({"tensor": None}) is None


def test_tensor_payload_is_much_smaller_than_legacy(tmp_path):
    arr = _rng_array(seed=3, shape=(3, 224, 288))
    legacy = tmp_path / "legacy.pt"
    _write_legacy(legacy, arr)
    cache = SmartCache(tmp_path / "cache", tqdm_enabled=False)
    p = _make_source(tmp_path, "big.bin", seed=9)
    cache.prepare([{**_sample(p), "bucket": (288, 224)}], lambda s: arr, num_workers=1)
    (new,) = _pt_files(tmp_path / "cache")
    ratio = new.stat().st_size / legacy.stat().st_size
    print(f"size ratio new/legacy = {ratio:.3f}")
    assert ratio < 0.8, ratio


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


def _pt_for(cache: SmartCache, path: str) -> Path:
    entry = cache._cache_index["entries"][path]
    return Path(cache._pt_path(entry["cache_file"], 0))


def _legacy_cache(tmp_path: Path, n_legacy: int, n_new: int):
    """Build ``n_legacy + n_new`` entries, then downgrade the first ``n_legacy``
    to the hand-written ndarray format. Returns the legacy/new file lists."""
    cache, samples, arrays = _build_cache(tmp_path, n=n_legacy + n_new)
    cache_dir = tmp_path / "cache"
    legacy_files = [_pt_for(cache, s["path"]) for s in samples[:n_legacy]]
    new_files = [_pt_for(cache, s["path"]) for s in samples[n_legacy:]]
    for pt, s in zip(legacy_files, samples[:n_legacy]):
        _write_legacy(pt, arrays[s["path"]], __source_path=s["path"])
    return cache, samples, arrays, cache_dir, legacy_files, new_files


def test_converter_converts_legacy_skips_new_and_is_idempotent(tmp_path):
    cache, samples, arrays, cache_dir, legacy, new = _legacy_cache(tmp_path, 3, 2)
    legacy_bytes = sum(p.stat().st_size for p in legacy)
    new_sizes = {p: p.stat().st_size for p in new}

    report = conv.convert_cache_dir(cache_dir, workers=2)
    assert report.converted == 3 and report.skipped == 2 and report.failed == 0
    assert report.bytes_before == legacy_bytes + sum(new_sizes.values())
    assert report.bytes_after < report.bytes_before
    assert report.converted_bytes_before == legacy_bytes
    assert report.converted_bytes_after < legacy_bytes

    for pt in legacy + new:
        raw = torch.load(pt, weights_only=False, map_location="cpu")
        assert isinstance(raw["tensor"], torch.Tensor)
    for pt, size in new_sizes.items():
        assert pt.stat().st_size == size  # untouched
    # Every entry still reads back bit-identically through the cache.
    for s in samples:
        got, _ = cache.get(s["path"])
        assert np.array_equal(got.numpy(), arrays[s["path"]])
    # No temp/partial files left behind.
    assert _non_pt_leftovers(cache_dir) == []

    again = conv.convert_cache_dir(cache_dir, workers=1)
    assert again.converted == 0 and again.skipped == 5 and again.failed == 0
    assert again.bytes_before == again.bytes_after


def test_converter_dry_run_changes_nothing(tmp_path):
    _, _, _, cache_dir, legacy, new = _legacy_cache(tmp_path, 2, 1)
    before = {p: (p.stat().st_size, p.read_bytes()) for p in legacy + new}
    report = conv.convert_cache_dir(cache_dir, workers=2, dry_run=True)
    assert report.converted == 2 and report.skipped == 1 and report.dry_run is True
    assert report.bytes_after == report.bytes_before  # dry-run cannot know the after size
    for p, (size, data) in before.items():
        assert p.stat().st_size == size and p.read_bytes() == data
    assert _non_pt_leftovers(cache_dir) == []


def test_converter_preserves_metadata_keys_exactly(tmp_path):
    _, _, _, cache_dir, legacy, _ = _legacy_cache(tmp_path, 1, 0)
    (pt,) = legacy
    before = torch.load(pt, weights_only=False, map_location="cpu")
    conv.convert_cache_dir(cache_dir, workers=1)
    after = torch.load(pt, weights_only=False, map_location="cpu")
    assert set(before) == set(after)
    for k in before:
        if k != "tensor":
            assert before[k] == after[k], k
    assert np.array_equal(after["tensor"].numpy(), before["tensor"])


def test_converter_reports_and_skips_unreadable_file(tmp_path):
    _, _, _, cache_dir, legacy, _ = _legacy_cache(tmp_path, 1, 0)
    bad = cache_dir / "deadbeefdead_64x48_1.pt"
    bad.write_bytes(b"not a torch file")
    report = conv.convert_cache_dir(cache_dir, workers=1)
    assert report.converted == 1 and report.failed == 1
    assert bad.read_bytes() == b"not a torch file"
    assert _non_pt_leftovers(cache_dir) == []


def test_cli_main_dry_run(tmp_path, capsys):
    _, _, _, cache_dir, _, _ = _legacy_cache(tmp_path, 1, 1)
    rc = conv.main([str(cache_dir), "--dry-run", "--workers", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "converted" in out and "skipped" in out and "dry-run" in out


def test_cli_main_missing_dir(tmp_path):
    assert conv.main([str(tmp_path / "nope")]) != 0
