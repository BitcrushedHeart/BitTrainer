"""CPU unit tests for OFTv2 orthogonal fine-tuning (bittrainer/oft.py).

All tests run on CPU with dummy tensors — no GPU, no dataset, no training loop.
They cover the three orthogonalisation backends, the clipped-norm divergence
guard, identity initialisation, the adapter->full-weight merge equivalence, and
the freeze/trainable split.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bittrainer.group_trainer import GroupTrainConfig
from bittrainer.model import create_model
from bittrainer.oft import (
    OFTConv2d,
    OFTLinear,
    _largest_divisor_at_most,
    _OFTBase,
    merge_oft_into_model,
    merged_state_dict,
    oft_parameters,
    skew_to_rotation,
    wrap_backbone_with_oft,
)


def _eye(blocks: int, n: int, dtype=torch.float64) -> torch.Tensor:
    return torch.eye(n, dtype=dtype).expand(blocks, n, n).clone()


# --- skew_to_rotation: identity init & orthogonality ------------------------

@pytest.mark.parametrize("backend", ["cayley", "cayley_neumann", "cans"])
def test_zero_generator_is_identity(backend):
    a = torch.zeros(3, 4, 4, dtype=torch.float64)
    r, clipped = skew_to_rotation(a, backend=backend)
    assert not clipped
    torch.testing.assert_close(r, _eye(3, 4), atol=1e-9, rtol=0)


def test_cayley_is_exactly_orthogonal():
    torch.manual_seed(0)
    a = torch.randn(2, 5, 5, dtype=torch.float64)
    r, _ = skew_to_rotation(a, backend="cayley", clipped_norm=None)
    rrt = r @ r.transpose(-1, -2)
    torch.testing.assert_close(rrt, _eye(2, 5), atol=1e-9, rtol=0)


def test_cayley_neumann_orthogonal_in_valid_regime():
    # With a small skew norm the truncated Neumann series tracks the true inverse,
    # so R is approximately orthogonal.
    torch.manual_seed(1)
    a = 0.02 * torch.randn(2, 6, 6, dtype=torch.float64)
    r, clipped = skew_to_rotation(a, backend="cayley_neumann", clipped_norm=0.95, neumann_terms=6)
    assert not clipped
    rrt = r @ r.transpose(-1, -2)
    torch.testing.assert_close(rrt, _eye(2, 6), atol=1e-4, rtol=0)


def test_cans_beats_cayley_neumann_orthogonality():
    # At a moderate norm, the Newton-Schulz polar refinement (CANS) yields a more
    # orthogonal R than the truncated Neumann series alone.
    torch.manual_seed(2)
    a = 0.3 * torch.randn(1, 8, 8, dtype=torch.float64)
    eye = _eye(1, 8)
    r_n, _ = skew_to_rotation(a, backend="cayley_neumann", clipped_norm=0.95, neumann_terms=6)
    r_c, _ = skew_to_rotation(a, backend="cans", clipped_norm=0.95, neumann_terms=6, cans_iters=4)
    err_n = (r_n @ r_n.transpose(-1, -2) - eye).abs().max()
    err_c = (r_c @ r_c.transpose(-1, -2) - eye).abs().max()
    assert err_c < err_n


# --- clipped-norm divergence guard -----------------------------------------

def test_clip_engages_and_keeps_output_finite():
    # A large generator would push the Neumann series past its convergence radius;
    # the clip must fire and the result must stay finite.
    torch.manual_seed(3)
    a = 50.0 * torch.randn(2, 4, 4, dtype=torch.float64)
    r, clipped = skew_to_rotation(a, backend="cayley_neumann", clipped_norm=0.95, neumann_terms=6)
    assert clipped
    assert torch.isfinite(r).all()


def test_clip_disabled_does_not_flag():
    a = 50.0 * torch.randn(1, 4, 4, dtype=torch.float64)
    _, clipped = skew_to_rotation(a, backend="cans", clipped_norm=None)
    assert not clipped


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        skew_to_rotation(torch.zeros(1, 2, 2), backend="nope")


# --- block sizing ----------------------------------------------------------

@pytest.mark.parametrize("n,k,expected", [(80, 8, 8), (80, 7, 5), (7, 8, 7), (12, 5, 4), (13, 8, 1)])
def test_largest_divisor(n, k, expected):
    assert _largest_divisor_at_most(n, k) == expected


# --- OFTLinear: identity at init, merge equivalence -------------------------

def test_oftlinear_identity_at_init():
    torch.manual_seed(4)
    base = nn.Linear(16, 10)
    oft = OFTLinear(base, blocks=2).double()
    base = base.double()
    x = torch.randn(3, 16, dtype=torch.float64)
    torch.testing.assert_close(oft(x), base(x), atol=1e-9, rtol=0)


def test_oftlinear_merge_matches_forward():
    torch.manual_seed(5)
    base = nn.Linear(24, 12)
    oft = OFTLinear(base, blocks=3, backend="cans", cans_iters=4).double()
    with torch.no_grad():
        oft.oft_a.normal_(0, 0.1)
    x = torch.randn(4, 24, dtype=torch.float64)
    merged = oft.to_merged_linear()
    torch.testing.assert_close(oft(x), merged(x), atol=1e-9, rtol=0)


def test_oftlinear_freezes_base_weight():
    base = nn.Linear(8, 8)
    oft = OFTLinear(base, blocks=2)
    # base weight/bias live as buffers (not Parameters) => never trained.
    param_names = {n for n, _ in oft.named_parameters()}
    assert "oft_a" in param_names
    assert "base_weight" not in param_names


def test_dora_oft_optional():
    base = nn.Linear(8, 8)
    oft = OFTLinear(base, blocks=2, dora=True)
    assert oft.oft_m is not None
    names = {n for n, _ in oft.named_parameters()}
    assert "oft_m" in names


# --- wrapping a real ConvNeXt backbone -------------------------------------

def test_wrap_backbone_identity_and_freeze():
    model = create_model(model_size="atto", pretrained=False, num_classes=4).eval().double()
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    with torch.no_grad():
        before = model(x)
    n = wrap_backbone_with_oft(model, blocks=4)
    assert n > 0
    model.eval()
    with torch.no_grad():
        after = model(x)
    # Zero-init OFT => wrapped model is identical to the base model.
    torch.testing.assert_close(after, before, atol=1e-6, rtol=0)

    # Base backbone frozen; only OFT generators + head trainable.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(".oft_a" in n for n in trainable)
    assert any(n.startswith("head.") for n in trainable)
    assert not any(n.endswith("base_weight") for n in trainable)
    # A frozen backbone conv must not require grad.
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any("stages" in n for n in frozen)


def test_merged_state_dict_matches_vanilla_keys():
    vanilla = create_model(model_size="atto", pretrained=False, num_classes=4)
    wrapped = create_model(model_size="atto", pretrained=False, num_classes=4)
    wrapped.load_state_dict(vanilla.state_dict())
    wrap_backbone_with_oft(wrapped, blocks=4)
    merged_keys = set(merged_state_dict(wrapped).keys())
    vanilla_keys = set(vanilla.state_dict().keys())
    assert merged_keys == vanilla_keys


def test_merged_model_loads_into_vanilla():
    vanilla = create_model(model_size="atto", pretrained=False, num_classes=4)
    wrapped = create_model(model_size="atto", pretrained=False, num_classes=4)
    wrapped.load_state_dict(vanilla.state_dict())
    wrap_backbone_with_oft(wrapped, blocks=4)
    with torch.no_grad():
        for p in oft_parameters(wrapped):
            if p.dim() == 3:  # only perturb the oft_a generators
                p.normal_(0, 0.05)
    sd = merged_state_dict(wrapped)
    # The merged full-weight state_dict loads cleanly into a fresh vanilla model.
    fresh = create_model(model_size="atto", pretrained=False, num_classes=4)
    fresh.load_state_dict(sd)  # must not raise


def test_merge_into_model_restores_base_modules():
    model = create_model(model_size="atto", pretrained=False, num_classes=4)
    n = wrap_backbone_with_oft(model, blocks=4)
    assert n > 0
    # ConvNeXt V2 wraps as pointwise OFTConv2d (no nn.Linear in the backbone).
    assert any(isinstance(m, OFTConv2d) for m in model.modules())
    assert any(isinstance(m, _OFTBase) for m in model.modules())
    merged = merge_oft_into_model(model)
    assert not any(isinstance(m, _OFTBase) for m in merged.modules())


# --- config defaults --------------------------------------------------------

def test_config_defaults_are_fast_path():
    cfg = GroupTrainConfig(group_folder="x", num_classes=2, class_names=["a", "b"])
    assert cfg.oft_backend == "cayley_neumann"
    assert cfg.oft_clipped_norm == 0.95
    assert cfg.oft_dora is False
