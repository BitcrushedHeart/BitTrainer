"""Tests for bittrainer.multihead_ordinal — size parsing, volume index, band/volume mappings."""

from __future__ import annotations

from bittrainer.multihead_ordinal import (
    band_of,
    build_band_vocab,
    parse_size,
    size_to_band_index,
    size_to_volume,
    volume_index,
    volume_ranks,
)


def test_parse_size():
    assert parse_size("34DD") == (34, "DD")
    assert parse_size("28aaa") == (28, "AAA")
    assert parse_size("34Z") is None  # Z not a US cup
    assert parse_size("DD") is None


def test_sister_sizes_share_volume():
    # 32D = 34C = 36B (each band +2 with cup -1) -> equal volume.
    assert volume_index("32D") == volume_index("34C") == volume_index("36B")


def test_volume_monotonic_within_band():
    assert volume_index("34DD") > volume_index("34D") > volume_index("34C")


def test_band_of():
    assert band_of("34DD") == 34
    assert band_of("__none__") is None


def test_build_band_vocab_sorted_distinct():
    classes = ["__none__", "34DD", "32B", "34C", "28A"]
    assert build_band_vocab(classes) == ["28", "32", "34"]


def test_size_to_band_index_ignores_none():
    classes = ["__none__", "28A", "32B", "34DD"]
    vocab = build_band_vocab(classes)  # ["28","32","34"]
    idx = size_to_band_index(classes, vocab, ignore_index=-1)
    assert idx == [-1, 0, 1, 2]


def test_size_to_volume_none_sentinel():
    classes = ["__none__", "34C"]
    vols = size_to_volume(classes, none_value=-100.0)
    assert vols[0] == -100.0
    assert vols[1] == volume_index("34C")


def test_volume_ranks_sisters_share_rank():
    # 36B and 34C and 32D are sisters (same volume) -> same rank; 34DD is higher.
    classes = ["__none__", "32D", "34C", "36B", "34DD"]
    ranks = volume_ranks(classes, none_index=-1)
    assert ranks[0] == -1
    assert ranks[1] == ranks[2] == ranks[3]  # sisters share a rank
    assert ranks[4] > ranks[1]  # bigger volume -> higher rank
