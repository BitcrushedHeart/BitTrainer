"""Pair manifests, the group + source-identity aware split and the embedding
dataset for pairwise reward training (Bitcrush Optimise P8, ISSUE-1133).

A pair manifest is ``{"pairs": [{"chosen", "rejected", "caption", "group_id",
"type", "identities"}]}``. The split never lets a caption group, an image or a
shared source identity straddle train and val, is deterministic for a seed, and
records per-type counts so the trainer can report them. CPU only, no models.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from bittrainer.pair_dataset import (
    PairEmbeddingDataset,
    PairEmbeddings,
    PairRecord,
    group_aware_split,
    load_pair_manifest,
    load_split,
    per_type_counts,
    save_split,
    write_pair_manifest,
)


def _records(n_groups: int = 12, per_group: int = 2) -> list[PairRecord]:
    records = []
    for g in range(n_groups):
        for k in range(per_group):
            records.append(
                PairRecord(
                    chosen=f"/img/g{g}_c{k}.png",
                    rejected=f"/img/g{g}_r{k}.png",
                    caption=f"caption {g}",
                    group_id=f"group-{g}",
                    pair_type="normal" if g % 3 else "offpolicy",
                )
            )
    return records


def test_manifest_roundtrip_and_validation(tmp_path):
    records = _records(2, 1)
    records[0] = PairRecord(
        **{**records[0].__dict__, "identities": ("blake3:abc", "source:def")}
    )
    path = tmp_path / "pairs.json"
    write_pair_manifest(path, records, extra={"optimise": {"schema": 1}})
    data = json.loads(path.read_text("utf-8"))
    assert data["optimise"] == {"schema": 1}
    assert data["pairs"][0]["identities"] == ["blake3:abc", "source:def"]
    assert data["pairs"][0]["type"] == "offpolicy"
    loaded = load_pair_manifest(path)
    assert loaded == records

    (tmp_path / "bad.json").write_text(json.dumps({"pairs": [{"chosen": "x"}]}), "utf-8")
    with pytest.raises(ValueError, match="rejected"):
        load_pair_manifest(tmp_path / "bad.json")
    (tmp_path / "empty.json").write_text(json.dumps({"pairs": []}), "utf-8")
    with pytest.raises(ValueError, match="no pairs"):
        load_pair_manifest(tmp_path / "empty.json")


def test_per_type_counts():
    assert per_type_counts(_records(6, 2)) == {"normal": 8, "offpolicy": 4}


def _groups(records, indices):
    return {records[i].group_id for i in indices}


def _images(records, indices):
    out = set()
    for i in indices:
        out.add(records[i].chosen)
        out.add(records[i].rejected)
    return out


def test_split_keeps_groups_and_images_apart_and_is_deterministic():
    records = _records()
    split = group_aware_split(records, val_fraction=0.25, seed=7)
    assert sorted(split.train + split.val) == list(range(len(records)))
    assert not _groups(records, split.train) & _groups(records, split.val)
    assert not _images(records, split.train) & _images(records, split.val)
    assert 0 < len(split.val) < len(records)
    # Roughly the requested fraction (whole groups, so not exact).
    assert abs(len(split.val) / len(records) - 0.25) <= 2 / 12
    again = group_aware_split(records, val_fraction=0.25, seed=7)
    assert (again.train, again.val) == (split.train, split.val)
    other = group_aware_split(records, val_fraction=0.25, seed=8)
    assert (other.train, other.val) != (split.train, split.val)


def test_split_merges_groups_sharing_an_image_or_a_source_identity():
    records = _records(6, 1)
    # Group 1 reuses group 0's chosen image; group 3 shares a source photo
    # with group 4 (Off-Policy pairs built from one real image).
    records[1] = PairRecord(**{**records[1].__dict__, "chosen": records[0].chosen})
    records[3] = PairRecord(**{**records[3].__dict__, "identities": ("source:photo-9",)})
    records[4] = PairRecord(**{**records[4].__dict__, "identities": ("source:photo-9",)})
    for seed in range(6):
        split = group_aware_split(records, val_fraction=0.34, seed=seed)
        for side in (split.train, split.val):
            assert (0 in side) == (1 in side), seed
            assert (3 in side) == (4 in side), seed
        assert not _images(records, split.train) & _images(records, split.val)
    assert split.components >= 1
    assert split.held_out is True


def test_split_with_one_component_has_no_held_out_set():
    records = _records(3, 2)
    for r in range(1, len(records)):
        records[r] = PairRecord(**{**records[r].__dict__, "group_id": records[0].group_id})
    split = group_aware_split(records, val_fraction=0.2, seed=0)
    assert split.val == []
    assert split.held_out is False
    assert len(split.train) == len(records)


def test_split_persists_and_reloads_against_the_same_pairs(tmp_path):
    records = _records(5, 1)
    split = group_aware_split(records, val_fraction=0.4, seed=3)
    path = tmp_path / "split.json"
    save_split(path, split, records)
    data = json.loads(path.read_text("utf-8"))
    assert data["seed"] == 3 and data["val_fraction"] == 0.4
    assert data["train_groups"] and data["val_groups"]
    assert set(data["train_groups"]) | set(data["val_groups"]) == {r.group_id for r in records}
    loaded = load_split(path, records)
    assert (loaded.train, loaded.val) == (split.train, split.val)
    # A manifest that no longer matches the persisted split is refused.
    changed = records + [PairRecord("/x_c.png", "/x_r.png", "new", "group-new")]
    with pytest.raises(ValueError, match="split"):
        load_split(path, changed)


def test_embedding_dataset_yields_one_item_per_pair_with_equal_weight():
    records = _records(3, 2)
    dim = 4
    rng = np.random.default_rng(0)
    images = {p: rng.standard_normal(dim).astype(np.float32) for r in records for p in (r.chosen, r.rejected)}
    texts = {r.caption: rng.standard_normal(dim).astype(np.float32) for r in records}
    embeddings = PairEmbeddings(images=images, texts=texts, embed_dim=dim)
    dataset = PairEmbeddingDataset(records, embeddings)
    assert len(dataset) == len(records)
    i_c, i_r, t, type_index = dataset[4]
    assert torch.equal(i_c, torch.from_numpy(images[records[4].chosen]))
    assert torch.equal(i_r, torch.from_numpy(images[records[4].rejected]))
    assert torch.equal(t, torch.from_numpy(texts[records[4].caption]))
    assert dataset.type_names[type_index] == records[4].pair_type
    # Equal per-pair sampling: every pair appears exactly once per pass.
    tensors = dataset.tensors()
    assert tensors["chosen"].shape == (len(records), dim)
    assert tensors["types"].tolist().count(dataset.type_names.index("offpolicy")) == 2
    missing = PairEmbeddings(images={}, texts=texts, embed_dim=dim)
    with pytest.raises(KeyError):
        PairEmbeddingDataset(records, missing)
