"""FCMAE self-supervised pretraining tests (Bitcrush BitTrainer).

Pins the dense-masked FCMAE approximation and its GenericTrainer integration:

* ``make_random_masks`` produces an EXACT masked fraction of a 0/1 grid mask;
* the reconstruction loss is computed on MASKED patches only (visible patches
  contribute exactly zero) with per-patch pixel normalisation;
* the wrapped encoder+decoder forwards and back-propagates through the masking
  without in-place autograd breakage, and the decoder input is INVARIANT to
  masked-patch input pixels (the honest anti-leak claim);
* the end-to-end run exports a BARE-trunk safetensors (no ``decoder.``/
  ``backbone.`` keys) with clean provenance metadata that
  ``apply_backbone_init`` loads unchanged;
* pause / backup / resume round-trips and a fingerprint mismatch starts fresh;
* the deterministic held-out split and unreadable-file skipping behave.

CPU-only, ``atto`` backbone, 64px images, ``decoder_dim=64``,
``dataloader_workers=0``. All tests pass under ``CUDA_VISIBLE_DEVICES=-1``.
"""

from __future__ import annotations

import asyncio

import torch
from PIL import Image

import bittrainer.fcmae_trainer as ft
from bittrainer.fcmae_trainer import run_fcmae_pretraining


# --------------------------------------------------------------------------- #
# Scaffolding                                                                  #
# --------------------------------------------------------------------------- #


class _FlagEvent:
    """Duck-typed pause/stop event (``.is_set()``) a callback can flip."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def _make_images(root, n):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        path = root / f"img{i}.png"
        Image.new(
            "RGB", (72, 64), color=((i * 37) % 255, (i * 83) % 255, (i * 151) % 255)
        ).save(path)
        paths.append(str(path))
    return paths


def _request(
    tmp_path,
    *,
    n=6,
    epochs=1,
    steps_per_epoch=1,
    batch_size=2,
    mask_ratio=0.6,
    validation_split=0.0,
    with_hashes=False,
    **cfg,
):
    paths = _make_images(tmp_path / "imgs", n)
    candidate = tmp_path / "candidates" / "fcmae.safetensors"
    training_config = {
        "image_size": 64,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "mask_ratio": mask_ratio,
        "decoder_dim": 64,
        "validation_split": validation_split,
        "device": "cpu",
        "dataloader_workers": 0,
        # bf16 autocast on CPU is ~6.5x slower than fp32 for these tiny models;
        # the E2E tests pin lifecycle mechanics, not AMP, so run them fp32.
        "use_amp": False,
    }
    training_config.update(cfg)
    request = {
        "run_id": "run_fcmae",
        "family_name": "bitcrush_backbone",
        "architecture": "convnextv2",
        "size_alias": "lite",
        "display_size": "Lite",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(candidate),
        "images": paths,
        "training_config": training_config,
        "license_provenance": "locally_trained",
        "external_pretrained_used": False,
        "release_blocking": False,
    }
    if with_hashes:
        request["content_hashes"] = {
            p: f"{i:02d}" + "ab" * 31 for i, p in enumerate(paths)
        }
    return request


def _run(request, progress_callback=None):
    return asyncio.run(
        run_fcmae_pretraining(request, progress_callback=progress_callback)
    )


def _to_decoder_layout(patches, gh, gw):
    """(B, L, p*p*3) -> the decoder's (B, p*p*3, gh, gw) output layout."""
    b, _l, ppc = patches.shape
    return patches.transpose(1, 2).reshape(b, ppc, gh, gw)


def _normalized_target(images, patch_size):
    """Replicate ``fcmae_loss``'s per-patch pixel normalisation for the target."""
    target = ft._patchify(images.float(), patch_size)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    return (target - mean) / torch.sqrt(var + 1e-6)


# --------------------------------------------------------------------------- #
# 1. Mask geometry                                                            #
# --------------------------------------------------------------------------- #


def test_make_random_masks_exact_fraction():
    for gh, gw, ratio in [(2, 2, 0.6), (4, 4, 0.6), (4, 4, 0.75), (7, 5, 0.5)]:
        mask = ft.make_random_masks(3, gh, gw, ratio)
        assert mask.shape == (3, 1, gh, gw)
        # Only 0/1 values.
        assert torch.all((mask == 0) | (mask == 1))
        expected_masked = int(round(ratio * gh * gw))
        for b in range(3):
            zeros = int((mask[b] == 0).sum())
            assert zeros == expected_masked


# --------------------------------------------------------------------------- #
# 2. Loss is on the masked patches only                                       #
# --------------------------------------------------------------------------- #


def test_loss_ignores_visible_patches():
    torch.manual_seed(0)
    b, p, gh, gw = 2, 8, 2, 2
    size = p * gh
    ppc = p * p * 3
    images = torch.rand(b, 3, size, size)
    # 1 = visible, 0 = masked.
    grid_mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]]])
    masked = (1.0 - grid_mask).reshape(b, gh * gw)

    # pred equals the normalized target on masked patches, garbage on visible.
    target = _normalized_target(images, p)
    pred_patches = target.clone()
    for bi in range(b):
        for li in range(gh * gw):
            if masked[bi, li] == 0:  # visible
                pred_patches[bi, li] = 1000.0
    pred = _to_decoder_layout(pred_patches, gh, gw)
    loss = ft.fcmae_loss(pred, images, grid_mask, patch_size=p, norm_pix=True)
    assert abs(loss.item()) < 1e-6

    # Perturbing ONLY the visible patches of an arbitrary pred leaves loss fixed.
    pred2 = torch.randn(b, ppc, gh, gw)
    l1 = ft.fcmae_loss(pred2, images, grid_mask, patch_size=p, norm_pix=True)
    p3_patches = pred2.flatten(2).transpose(1, 2).clone()
    for bi in range(b):
        for li in range(gh * gw):
            if masked[bi, li] == 0:  # visible
                p3_patches[bi, li] += 500.0
    pred3 = _to_decoder_layout(p3_patches, gh, gw)
    l2 = ft.fcmae_loss(pred3, images, grid_mask, patch_size=p, norm_pix=True)
    assert torch.isclose(l1, l2)


# --------------------------------------------------------------------------- #
# 3. Forward / backward smoke test on CPU                                     #
# --------------------------------------------------------------------------- #


def _build_model(decoder_dim=64):
    from bittrainer.model import create_model

    encoder = create_model(model_size="atto", pretrained=False, num_classes=0)
    decoder = ft._FcmaeDecoder(
        encoder.num_features, decoder_dim=decoder_dim, patch_size=32
    )
    return ft._FcmaeModel(encoder, decoder, patch_size=32)


def test_forward_backward_smoke_cpu():
    torch.manual_seed(0)
    model = _build_model()
    model.train()
    images = torch.randn(2, 3, 64, 64)
    grid_mask = ft.make_random_masks(2, 2, 2, 0.6)
    pred = model.forward_masked(images, grid_mask)
    assert pred.shape == (2, 32 * 32 * 3, 2, 2)
    loss = ft.fcmae_loss(pred, images, grid_mask, patch_size=32)
    assert torch.isfinite(loss)
    loss.backward()

    stem_grad = model.encoder.stem[0].weight.grad
    assert stem_grad is not None and stem_grad.abs().sum() > 0
    token_grad = model.decoder.mask_token.grad
    assert token_grad is not None and token_grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# 4. Masked-patch input pixels never reach the decoder                        #
# --------------------------------------------------------------------------- #


def test_feature_masking_blocks_leakage():
    torch.manual_seed(0)
    model = _build_model().eval()
    images = torch.randn(1, 3, 64, 64)
    # Patch (0, 0) masked, the rest visible.
    grid_mask = torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]])

    with torch.no_grad():
        ref = model.forward_masked(images, grid_mask)

    # Changing a MASKED patch's input pixels leaves the output unchanged: the
    # masked patch is zeroed at the input, so its content never enters.
    masked_changed = images.clone()
    masked_changed[:, :, 0:32, 0:32] += 5.0
    with torch.no_grad():
        out_masked = model.forward_masked(masked_changed, grid_mask)
    assert torch.allclose(ref, out_masked)

    # Changing a VISIBLE patch's input pixels DOES change the output.
    visible_changed = images.clone()
    visible_changed[:, :, 0:32, 32:64] += 5.0
    with torch.no_grad():
        out_visible = model.forward_masked(visible_changed, grid_mask)
    assert not torch.allclose(ref, out_visible)


# --------------------------------------------------------------------------- #
# 5. End-to-end pretrain: export contract + apply_backbone_init roundtrip     #
# --------------------------------------------------------------------------- #


def test_end_to_end_export_and_backbone_init_load(tmp_path):
    """One lifecycle run pins both the export contract and its consumption:
    bare-trunk keys + provenance metadata, then apply_backbone_init loads the
    trunk into a fresh supervised model (the fine-tune path)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    from bittrainer.backbone_init import apply_backbone_init
    from bittrainer.model import create_model

    request = _request(
        tmp_path,
        n=8,
        epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        validation_split=0.25,
        with_hashes=True,
    )
    result = _run(request)
    assert result["candidate_checkpoint_path"] == request["candidate_checkpoint_path"]
    assert result["steps_completed"] > 0

    with safe_open(result["candidate_checkpoint_path"], framework="pt") as f:
        keys = list(f.keys())
        metadata = f.metadata()

    assert not any(k.startswith("decoder.") for k in keys)
    assert not any(k.startswith("backbone.") for k in keys)
    assert "stem.0.weight" in keys
    assert metadata["pretrain_method"] == "fcmae"
    assert metadata["mask_ratio"] == "0.6"
    assert metadata["external_pretrained_used"] == "false"
    assert metadata["license_provenance"] == "locally_trained"
    assert metadata["dataset_image_count"] == "8"

    model = create_model(model_size="atto", pretrained=False, num_classes=2)
    loaded = apply_backbone_init(
        model,
        {
            "source": "local_candidate",
            "checkpoint_path": result["candidate_checkpoint_path"],
        },
    )
    assert loaded is True
    saved = load_file(result["candidate_checkpoint_path"])
    assert torch.equal(model.state_dict()["stem.0.weight"], saved["stem.0.weight"])


# --------------------------------------------------------------------------- #
# 7. Backup + resume round-trip                                               #
# --------------------------------------------------------------------------- #


def test_backup_resume_roundtrip(tmp_path):
    from bittrainer.training_state import TrainingStateManager

    backup_dir = str(tmp_path / "backups")

    pause = _FlagEvent()

    def cb_a(msg):
        if msg.get("stage") == "training" and msg.get("epoch") == 1:
            pause.set()

    request_a = _request(
        tmp_path,
        n=4,
        epochs=2,
        steps_per_epoch=1,
        batch_size=2,
        validation_split=0.0,
        backup_dir=backup_dir,
    )
    result_a = asyncio.run(
        run_fcmae_pretraining(request_a, progress_callback=cb_a, pause_event=pause)
    )
    assert result_a.get("paused") is True
    assert TrainingStateManager(backup_dir).list_backups()
    steps_at_pause = result_a["global_step"]

    request_b = _request(
        tmp_path,
        n=4,
        epochs=2,
        steps_per_epoch=1,
        batch_size=2,
        validation_split=0.0,
        backup_dir=backup_dir,
    )
    request_b["training_config"]["resume_from"] = backup_dir
    events = []
    result_b = asyncio.run(
        run_fcmae_pretraining(request_b, progress_callback=events.append)
    )
    resumed = [m for m in events if m.get("type") == "training_resumed"]
    assert resumed and resumed[0]["epoch"] == 1
    assert resumed[0]["global_step"] > 0
    assert "paused" not in result_b
    assert result_b["epochs_completed"] == 2
    assert result_b["steps_completed"] > steps_at_pause


# --------------------------------------------------------------------------- #
# 8. Fingerprint mismatch starts fresh                                        #
# --------------------------------------------------------------------------- #


def test_fingerprint_mismatch_starts_fresh(tmp_path):
    backup_dir = str(tmp_path / "backups")

    pause = _FlagEvent()

    def cb_a(msg):
        if msg.get("stage") == "training" and msg.get("epoch") == 1:
            pause.set()

    request_a = _request(
        tmp_path,
        n=4,
        epochs=2,
        steps_per_epoch=1,
        batch_size=2,
        validation_split=0.0,
        mask_ratio=0.6,
        backup_dir=backup_dir,
    )
    result_a = asyncio.run(
        run_fcmae_pretraining(request_a, progress_callback=cb_a, pause_event=pause)
    )
    assert result_a.get("paused") is True

    request_b = _request(
        tmp_path,
        n=4,
        epochs=2,
        steps_per_epoch=1,
        batch_size=2,
        validation_split=0.0,
        mask_ratio=0.75,
        backup_dir=backup_dir,
    )
    request_b["training_config"]["resume_from"] = backup_dir
    events = []
    result_b = asyncio.run(
        run_fcmae_pretraining(request_b, progress_callback=events.append)
    )
    assert any(
        m.get("type") == "resume_skipped" and m.get("reason") == "fingerprint_mismatch"
        for m in events
    )
    assert "paused" not in result_b
    # A fresh start never emits a resumed frame.
    assert not any(m.get("type") == "training_resumed" for m in events)


# --------------------------------------------------------------------------- #
# 9. Deterministic held-out split                                             #
# --------------------------------------------------------------------------- #


def test_holdout_split_deterministic():
    paths = [f"/data/img_{i:04d}.png" for i in range(200)]

    train1, val1 = ft._split_paths(paths, {}, 0.25, 1000)
    train2, val2 = ft._split_paths(paths, {}, 0.25, 1000)
    assert train1 == train2 and val1 == val2

    assert set(train1).isdisjoint(val1)
    assert set(train1) | set(val1) == set(paths)

    fraction = len(val1) / len(paths)
    assert 0.5 * 0.25 <= fraction <= 2.0 * 0.25

    _, val_trunc1 = ft._split_paths(paths, {}, 0.25, 10)
    _, val_trunc2 = ft._split_paths(paths, {}, 0.25, 10)
    assert val_trunc1 == val_trunc2
    assert len(val_trunc1) <= 10
    assert set(val_trunc1).issubset(set(val1))


# --------------------------------------------------------------------------- #
# 10. Unreadable file skipped                                                 #
# --------------------------------------------------------------------------- #


def test_unreadable_file_skipped(tmp_path):
    from pathlib import Path

    # steps_per_epoch=0 = one FULL pass, so the bad file is GUARANTEED to be
    # drawn (a step-capped epoch might shuffle it out of the consumed slice).
    request = _request(tmp_path, n=4, epochs=1, steps_per_epoch=0, validation_split=0.0)
    bad = tmp_path / "imgs" / "broken.png"
    bad.write_text("this is definitely not a PNG")
    request["images"].append(str(bad))

    result = _run(request)  # completes without raising
    assert result["steps_completed"] > 0
    assert Path(result["candidate_checkpoint_path"]).is_file()
    # The bad file was gathered (has a .png suffix) but dropped at load time.
    assert result["dataset_image_count"] == 5
