"""Training images are not mirrored, and labels that depend on geometry are
handled gently (Bitcrush ISSUE-1655).

The backbone's training transform flipped half of all images, whatever they
were labelled. A mirror changes the answer for direction, roll and position
groups, so half the training images of such a group carried a wrong label. The
flip is now off unless ``hflip`` asks for it.

The other geometric augmentations do the same on a smaller scale: the random
crop reframes an image, and RandAugment may rotate it by 9 degrees, shear it or
shift it by 13%. A request can list the heads whose answer depends on framing,
position or tilt (``geometry_sensitive``; a pack carries the list in its
manifest). An image with a human label for one of them is prepared like a
validation image and gets the colour jitter alone. Every other image keeps the
usual augmentation, and validation is as it was.

Generated images; no training run.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import torch
from PIL import Image

import bittrainer.backbone_trainer as bb

_TODAY = [
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "RandAugment",
    "ColorJitter",
    "ToTensor",
    "Normalize",
    "RandomErasing",
]
_NO_FLIP = [name for name in _TODAY if name != "RandomHorizontalFlip"]
_VALIDATION = ["Resize", "CenterCrop", "ToTensor", "Normalize"]
# The flip and the colour jitter alone: no crop, no RandAugment, no erasing.
_PLAIN = {"use_random_resized_crop": False, "randaugment_n": 0, "random_erasing_p": 0.0}
# The random crop and the colour jitter alone.
_CROP_ONLY = {"randaugment_n": 0, "random_erasing_p": 0.0}


def _names(pipeline) -> list[str]:
    return [type(step).__name__ for step in pipeline.transforms]


def _two_tone(folder, width: int, height: int) -> str:
    """An image that is red on its left three eighths and blue on the rest."""
    image = Image.new("RGB", (width, height), (20, 40, 160))
    image.paste((200, 60, 30), (0, 0, width * 3 // 8, height))
    path = folder / f"two_tone_{width}x{height}.png"
    image.save(path)
    return str(path)


@pytest.fixture
def wide(tmp_path) -> str:
    """Twice as wide as tall: a centre crop and a squash frame it differently."""
    return _two_tone(tmp_path, 64, 32)


@pytest.fixture
def square(tmp_path) -> str:
    return _two_tone(tmp_path, 64, 64)


def _red(tensor: torch.Tensor) -> torch.Tensor:
    """Where the output is red and not blue: the picture's geometry, whatever
    the colour jitter did to its colours."""
    return tensor[0] > tensor[2]


def _red_is_on_the_left(tensor: torch.Tensor) -> bool:
    red = _red(tensor).float()
    half = red.shape[-1] // 2
    return bool(red[:, :half].mean() > red[:, half:].mean())


def _draws(path: str, aug: dict, binary=None, groups=None, *, count: int = 200):
    """The same sample fetched ``count`` times from a training dataset."""
    sample = bb._Sample(path, binary or {}, groups or {})
    dataset = bb._BackboneDataset([sample], bb._train_transform(16, aug))
    torch.manual_seed(0)
    return [dataset[0] for _ in range(count)]


def _as_validation(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        return bb._val_transform(16)(image.convert("RGB"))


# -- no mirroring -------------------------------------------------------------


def test_the_default_pipeline_has_no_flip_and_the_knob_brings_it_back():
    assert _names(bb._train_transform(64)) == _NO_FLIP
    assert _names(bb._train_transform(64, {"hflip": False})) == _NO_FLIP
    assert _names(bb._train_transform(64, {"hflip": True})) == _TODAY


def test_no_image_is_mirrored_by_default(wide):
    draws = _draws(wide, _PLAIN)

    assert all(_red_is_on_the_left(tensor) for tensor, _b, _g in draws)


def test_the_knob_mirrors_half_the_images_again(wide):
    draws = _draws(wide, {**_PLAIN, "hflip": True})

    kept = sum(_red_is_on_the_left(tensor) for tensor, _b, _g in draws)
    assert 0.4 < kept / len(draws) < 0.6


# -- labels that depend on geometry -------------------------------------------

_SENSITIVE = ["roll", "on_left"]


def test_an_image_labelled_for_a_listed_head_gets_colour_only(wide):
    aug = {"geometry_sensitive": _SENSITIVE}  # every other knob at its default
    by_group = _draws(wide, aug, groups={"roll": 1}, count=40)
    by_head = _draws(wide, aug, binary={"on_left": 1.0, "plain": 0.0}, count=40)

    reference = _as_validation(wide)
    for draws in (by_group, by_head):
        tensors = [tensor for tensor, _b, _g in draws]
        # Framed exactly as validation frames it, every time ...
        assert all(torch.equal(_red(tensor), _red(reference)) for tensor in tensors)
        # ... while its colours still vary from draw to draw.
        assert any(not torch.equal(tensor, tensors[0]) for tensor in tensors)
        assert all(not torch.equal(tensor, reference) for tensor in tensors)


def _framings(draws) -> int:
    """How many different ways the draws framed the picture."""
    return len({_red(tensor).numpy().tobytes() for tensor, _b, _g in draws})


def test_an_image_without_such_a_label_still_gets_the_random_crop(square):
    aug = {**_CROP_ONLY, "geometry_sensitive": _SENSITIVE}

    unlisted = _draws(square, aug, binary={"plain": 1.0}, groups={"style": 0}, count=40)
    label_free = _draws(square, aug, count=40)  # what a distillation run adds
    listed = _draws(square, aug, groups={"roll": 0}, count=40)

    assert _framings(unlisted) > 3
    assert _framings(label_free) > 3
    assert _framings(listed) == 1


def test_an_implied_negative_is_not_a_label_for_this(square):
    """Which images keep the full augmentation must not change with the label
    policy being compared."""
    aug = {**_CROP_ONLY, "geometry_sensitive": _SENSITIVE}

    implied = {"on_left": bb._ImpliedNegative(0.1), "plain": 1.0}
    draws = _draws(square, aug, binary=implied, count=40)

    assert _framings(draws) > 3


def test_the_labels_come_back_as_they_went_in(wide):
    binary, groups = {"on_left": 1.0}, {"roll": 0}

    for _tensor, got_binary, got_groups in _draws(
        wide, {"geometry_sensitive": _SENSITIVE}, binary, groups, count=5
    ):
        assert got_binary is binary and got_groups is groups


def test_names_are_matched_the_way_heads_are_named():
    """A dot in a name becomes an underscore in its head's name."""
    transform = bb._train_transform(16, {"geometry_sensitive": ["A. Roll"]})

    assert transform.sensitive == {"A_ Roll"}


def test_without_a_list_the_pipeline_is_one_plain_compose():
    for aug in (None, {}, {"geometry_sensitive": None}, {"geometry_sensitive": []}):
        assert _names(bb._train_transform(64, aug)) == _NO_FLIP


def test_with_a_list_the_two_pipelines_are_the_usual_and_validation_plus_jitter():
    transform = bb._train_transform(64, {"geometry_sensitive": _SENSITIVE})

    assert isinstance(transform, bb._GeometryCarefulTransform)
    assert _names(transform.usual) == _NO_FLIP
    assert _names(transform.colour_only) == [
        "Resize",
        "CenterCrop",
        "ColorJitter",
        "ToTensor",
        "Normalize",
    ]


def test_validation_is_unchanged():
    assert _names(bb._val_transform(64)) == _VALIDATION


# -- from the request and the pack to the dataset -----------------------------


def _request(tmp_path, **config) -> dict:
    records = []
    for index in range(6):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (16, 16), (index * 40, 0, 0)).save(path)
        records.append(
            {
                # Spread over the hash split: one image validates, five train.
                "content_hash": f"{index * 1777 + 2000:08x}",
                "file_paths": [str(path)],
                "binary": {"on_left": "positive" if index % 2 else "negative"},
                "groups": {"roll": ("left", "level", "right")[index % 3]},
            }
        )
    return {
        "run_id": "run_geometry",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(tmp_path / "candidate.safetensors"),
        "records": records,
        "training_config": {
            "device": "cpu",
            "image_size": 16,
            "batch_size": 2,
            "use_amp": False,
            **config,
        },
    }


@pytest.fixture
def loaders(monkeypatch):
    """Run ``run_backbone_training`` up to where its loaders are built, and
    hand back their transforms."""
    import bittrainer.generic.generic_trainer as generic

    seen: dict = {}

    def _run(self, task, **_kwargs):
        context = task.make_context(None, None, None, None)
        train_loader, *_rest = task.build_loaders(context, 0, task.batch_size, None)
        seen["train"] = train_loader.dataset.transform
        seen["validation"] = task._val_loader.dataset.transform
        return {}

    monkeypatch.setattr(generic.GenericTrainer, "run", _run)

    def _transforms(request: dict) -> dict:
        assert asyncio.run(bb.run_backbone_training(request)) == {}
        return dict(seen)

    return _transforms


def test_a_requests_list_reaches_the_training_dataset(tmp_path, loaders):
    request = _request(tmp_path)
    request["geometry_sensitive"] = _SENSITIVE

    seen = loaders(request)

    assert isinstance(seen["train"], bb._GeometryCarefulTransform)
    assert seen["train"].sensitive == set(_SENSITIVE)
    assert _names(seen["train"].usual) == _NO_FLIP
    assert _names(seen["validation"]) == _VALIDATION


def test_a_request_without_a_list_trains_without_the_flip_and_nothing_else(
    tmp_path, loaders
):
    seen = loaders(_request(tmp_path))

    assert _names(seen["train"]) == _NO_FLIP
    assert _names(seen["validation"]) == _VALIDATION


def test_the_flip_can_be_asked_for_in_the_training_settings(tmp_path, loaders):
    seen = loaders(_request(tmp_path, hflip=True))

    assert _names(seen["train"]) == _TODAY
    assert _names(seen["validation"]) == _VALIDATION


def test_a_pack_carries_the_list_in_its_manifest(tmp_path):
    from bittrainer.backbone_pack import PackError, pack_geometry_sensitive

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": 1}), encoding="utf-8")
    assert pack_geometry_sensitive(tmp_path) == []

    listed = {"format": 1, "geometry_sensitive": ["g3", "c12", "m57"]}
    manifest.write_text(json.dumps(listed), encoding="utf-8")
    assert pack_geometry_sensitive(tmp_path) == ["g3", "c12", "m57"]

    manifest.write_text(
        json.dumps({"format": 1, "geometry_sensitive": "g3"}), encoding="utf-8"
    )
    with pytest.raises(PackError):
        pack_geometry_sensitive(tmp_path)


def test_the_runner_copies_a_packs_list_into_the_request(tmp_path):
    import bittrainer.tools.backbone_pack_train as runner
    from bittrainer.tests.test_backbone_pack_train import _make_pack, _options

    pack = _make_pack(tmp_path)
    plain, _counts = runner.build_request(pack, tmp_path / "run", **_options())
    assert "geometry_sensitive" not in plain

    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    manifest["geometry_sensitive"] = ["g5", "c12"]
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    request, _counts = runner.build_request(pack, tmp_path / "run", **_options())
    assert request["geometry_sensitive"] == ["g5", "c12"]
