"""Region-crop mechanism: generalising the face-crop pre-pass.

The face path stores a union bbox of ALL detections; region crops need a
class filter (a body-part detector may emit several classes) and a
highest-confidence selection mode, plus cache identities that distinguish
model/classes/selection so switching the crop never reuses stale tensors.
`face_aware_crop` itself is bbox-generic and stays untouched.
"""

from __future__ import annotations

from bittrainer.face_crop import region_bbox_cache_name, select_bbox
from bittrainer.smart_cache import face_model_signature, region_signature


class TestSelectBbox:
    DETS = [
        ("face", 0.9, (10, 10, 20, 20)),
        ("crotch", 0.7, (40, 50, 60, 80)),
        ("crotch", 0.85, (42, 52, 58, 78)),
        ("hand", 0.95, (0, 0, 5, 5)),
    ]

    def test_union_over_all_when_no_filter(self) -> None:
        assert select_bbox(self.DETS) == [0, 0, 60, 80]

    def test_class_filter_unions_only_matches(self) -> None:
        assert select_bbox(self.DETS, target_classes=["crotch"]) == [40, 50, 60, 80]

    def test_highest_conf_picks_single_best_match(self) -> None:
        got = select_bbox(self.DETS, target_classes=["crotch"], selection="highest_conf")
        assert got == [42, 52, 58, 78]

    def test_filter_is_case_insensitive(self) -> None:
        assert select_bbox(self.DETS, target_classes=["Crotch"]) == [40, 50, 60, 80]

    def test_no_match_returns_empty(self) -> None:
        assert select_bbox(self.DETS, target_classes=["areola"]) == []

    def test_empty_detections_return_empty(self) -> None:
        assert select_bbox([]) == []
        assert select_bbox([], target_classes=["crotch"]) == []


class TestRegionCacheName:
    def test_deterministic(self) -> None:
        a = region_bbox_cache_name("BMD_Crotch_s.pt", ["crotch"], "highest_conf")
        b = region_bbox_cache_name("BMD_Crotch_s.pt", ["crotch"], "highest_conf")
        assert a == b
        assert a.startswith("region_bboxes_")
        assert a.endswith(".json")

    def test_distinct_by_model_classes_and_selection(self) -> None:
        base = region_bbox_cache_name("BMD_Crotch_s.pt", ["crotch"], "highest_conf")
        assert region_bbox_cache_name("BMD_Labia_s.pt", ["crotch"], "highest_conf") != base
        assert region_bbox_cache_name("BMD_Crotch_s.pt", ["labia"], "highest_conf") != base
        assert region_bbox_cache_name("BMD_Crotch_s.pt", ["crotch"], "union") != base

    def test_class_order_does_not_matter(self) -> None:
        a = region_bbox_cache_name("m.pt", ["a", "b"], "union")
        b = region_bbox_cache_name("m.pt", ["b", "a"], "union")
        assert a == b


class TestRegionSignature:
    def test_face_parity_for_default_args(self, tmp_path) -> None:
        # Existing face-crop groups must produce byte-identical signatures so
        # the change cannot invalidate their baked tensor caches.
        model = tmp_path / "BMDSeg_Face_s.pt"
        model.write_bytes(b"weights")
        assert region_signature(str(model)) == face_model_signature(str(model))
        assert region_signature(str(model), None, "union") == face_model_signature(str(model))
        assert region_signature(None) == face_model_signature(None)

    def test_classes_and_selection_change_the_signature(self, tmp_path) -> None:
        model = tmp_path / "BMD_Crotch_s.pt"
        model.write_bytes(b"weights")
        base = region_signature(str(model))
        with_classes = region_signature(str(model), ["crotch"], "union")
        with_selection = region_signature(str(model), ["crotch"], "highest_conf")
        assert with_classes != base
        assert with_selection != with_classes


class TestDropPathsWithoutBbox:
    def _make_group(self, tmp_path, n_a: int = 4, n_b: int = 3):
        from PIL import Image

        for cls, n in (("alpha", n_a), ("beta", n_b)):
            d = tmp_path / cls / "train"
            d.mkdir(parents=True)
            for i in range(n):
                Image.new("RGB", (32, 32), (i * 30, 0, 0)).save(d / f"img_{i}.jpg")
        return tmp_path

    def test_drop_survives_reshuffle(self, tmp_path) -> None:
        from bittrainer.group_dataset import GroupDataset

        group = self._make_group(tmp_path)
        # natural_sampling: one sample per image, no replicated draws — the
        # test asserts on exact path sets.
        ds = GroupDataset(group, ["alpha", "beta"], split="train", natural_sampling=True)
        all_paths = sorted({s["path"] for s in ds.samples})
        assert len(all_paths) == 7

        # Region found on all but two images.
        bboxes = {p: [1, 1, 5, 5] for p in all_paths[:-2]}
        dropped = ds.drop_paths_without_bbox(bboxes)
        assert dropped == 2
        assert {s["path"] for s in ds.samples} == set(all_paths[:-2])

        ds.reshuffle()  # rebuilds samples from the path lists
        assert {s["path"] for s in ds.samples} == set(all_paths[:-2])

    def test_noop_when_all_have_bboxes(self, tmp_path) -> None:
        from bittrainer.group_dataset import GroupDataset

        group = self._make_group(tmp_path)
        ds = GroupDataset(group, ["alpha", "beta"], split="train", natural_sampling=True)
        bboxes = {s["path"]: [1, 1, 5, 5] for s in ds.samples}
        assert ds.drop_paths_without_bbox(bboxes) == 0
        assert len({s["path"] for s in ds.samples}) == 7
