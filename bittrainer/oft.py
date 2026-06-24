"""OFTv2-style orthogonal fine-tuning for ConvNeXt V2 backbones.

Orthogonal Fine-Tuning (OFT) freezes a layer's base weight ``W`` and learns a
*block-diagonal orthogonal* rotation ``R`` applied on the output side, so the
adapted weight is ``W' = R @ W``. Because ``R`` is orthogonal it preserves the
pairwise angles (and norms) of the weight rows, which empirically resists the
catastrophic forgetting that unconstrained fine-tuning can cause on small
incremental datasets — the property we want for the "Orthogonal / Fast
Incremental" training rung.

timm's ConvNeXt V2 implements every block as ``Conv2d`` (the pointwise MLP layers
``mlp.fc1``/``mlp.fc2`` are 1×1 convolutions; ``conv_dw`` is a depthwise 7×7
conv). OFT targets the **pointwise 1×1 convs** — their weight ``[out, in, 1, 1]``
is exactly a channel-wise linear map, so the same ``W' = R @ W`` rotation applies
once the trailing singleton kernel dims are squeezed. The depthwise convs, the
4×4 stem, and the 2×2 downsample convs are left frozen (a depthwise conv is
per-channel, not a channel mixer, so a channel rotation is ill-defined there).
``nn.Linear`` layers are also targeted for completeness/portability.

``R`` is parameterised by a learnable per-block generator ``A`` via the skew
matrix ``Q = A - Aᵀ`` (so ``Q`` is skew-symmetric and ``Q`` -> ``R`` keeps ``R``
in the special-orthogonal group). ``A`` initialises to **zero**, so ``Q = 0`` and
``R = I`` — i.e. an untrained OFT layer is exactly the warm-started base model,
which keeps the start-of-training checkpoint identical to the incumbent.

Three orthogonalisation backends (see OneTrainer references in the suite issue):

* ``cayley`` — exact Cayley transform ``R = (I + Q)⁻¹ (I − Q)`` via a linear
  solve. No approximation, but a matrix solve per block per step.
* ``cayley_neumann`` (**default**) — approximates ``(I + Q)⁻¹`` with a truncated
  Neumann series ``Σ (−Q)ᵏ``. Cheap (matmuls only, no solve) but the series only
  converges while ``‖Q‖ < 1``; at ``‖Q‖ ≥ 1`` it diverges into garbage gradients.
  Guarded by **clipped OFT norm** (``oft_clipped_norm``, default ``0.95``): each
  block's ``Q`` is scaled back under the threshold before the series. We clip the
  Frobenius norm, which upper-bounds the spectral radius, so convergence is
  guaranteed conservatively. (OneTrainer PR #1492.)
* ``cans`` — Chebyshev-optimised Newton–Schulz: builds the Neumann approximation
  then refines it onto the orthogonal manifold with Newton–Schulz polar
  iterations. More matmuls, lower orthogonalisation error; the "willing to wait"
  quality option. Does not require the Neumann clip for convergence, but the clip
  is still applied if configured, for consistency. (OneTrainer PR #1512.)

``oft_dora`` enables an **experimental** DoRA-OFT variant (OFT rotation + a
learned per-output magnitude); it is never the default and is not exposed in the
main UI. (OneTrainer PR #1335.)

The adapter is always **merged back into full weights** before checkpointing
(:func:`merged_state_dict`), so the artefact a training run promotes is a plain
ConvNeXt ``state_dict`` — identical in format to a full fine-tune, loadable by
``model.load_checkpoint`` unchanged, and correctly reflected by
``model.backbone_feature_hash``.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("cayley", "cayley_neumann", "cans")

# Head modules are never wrapped — OFT adapts the frozen backbone only; the head
# tail is trained directly (like the cached-feature probe).
_HEAD_PREFIX = "head."


def _largest_divisor_at_most(n: int, k: int) -> int:
    """Largest divisor of ``n`` that is ``<= k`` (and ``>= 1``).

    OFT needs the output dimension split into equal blocks; when the requested
    block count does not divide ``n`` we fall back to the nearest coarser split
    that does, so block sizes stay uniform without padding.
    """
    k = max(1, min(k, n))
    for d in range(k, 0, -1):
        if n % d == 0:
            return d
    return 1


def _clip_skew(q: torch.Tensor, clipped_norm: float | None) -> tuple[torch.Tensor, bool]:
    """Scale each block's skew matrix so its Frobenius norm stays under threshold.

    ``q`` is ``[blocks, bs, bs]``. Returns the (possibly) clipped tensor and a
    flag indicating whether any block was clipped. ``clipped_norm=None`` disables
    clipping (only safe for backends that do not rely on Neumann convergence).
    """
    if clipped_norm is None:
        return q, False
    norms = torch.linalg.matrix_norm(q, ord="fro")  # [blocks]
    over = norms > clipped_norm
    if not bool(over.any()):
        return q, False
    scale = torch.ones_like(norms)
    scale[over] = clipped_norm / norms[over]
    return q * scale.view(-1, 1, 1), True


def _neumann_inverse(q: torch.Tensor, terms: int) -> torch.Tensor:
    """Approximate ``(I + Q)⁻¹`` as the truncated Neumann series ``Σ (−Q)ᵏ``.

    ``terms`` is the highest power retained (``terms=0`` -> identity only).
    Accumulated iteratively so each added term costs one batched matmul.
    """
    eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device).expand_as(q)
    acc = eye.clone()
    term = eye.clone()
    for _ in range(terms):
        term = -term @ q
        acc = acc + term
    return acc


def _newton_schulz_orthogonalize(r: torch.Tensor, iters: int) -> torch.Tensor:
    """Refine ``r`` onto the orthogonal manifold via Newton–Schulz polar iteration.

    Uses the cubic iteration ``X <- 1.5 X − 0.5 X (Xᵀ X)`` (Higham), which drives
    the singular values of ``X`` to 1 — i.e. toward the orthogonal polar factor —
    when they start within ``(0, √3)``. The Cayley/Neumann seed is already close
    to orthogonal, so a couple of iterations remove the residual orthogonality
    error left by the truncated series (the "Chebyshev-optimised" framing: a short
    fixed schedule of polar steps tuned for fast terminal convergence).
    """
    for _ in range(iters):
        rtr = r.transpose(-1, -2) @ r
        r = 1.5 * r - 0.5 * (r @ rtr)
    return r


def skew_to_rotation(
    a: torch.Tensor,
    *,
    backend: str = "cayley_neumann",
    clipped_norm: float | None = 0.95,
    neumann_terms: int = 6,
    cans_iters: int = 3,
) -> tuple[torch.Tensor, bool]:
    """Map a per-block generator ``A`` to a block-diagonal rotation ``R``.

    ``a`` is ``[blocks, bs, bs]``. Returns ``(R, clipped)`` where ``R`` has the
    same shape and ``clipped`` reports whether the Neumann norm guard fired.
    With ``A = 0`` the result is exactly the identity, regardless of backend.
    """
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown oft_backend '{backend}'. Valid: {VALID_BACKENDS}")

    q = a - a.transpose(-1, -2)  # skew-symmetric
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand_as(a)

    if backend == "cayley":
        # Exact: R = (I + Q)^{-1} (I - Q). Clip is irrelevant to the solve but
        # applied when configured so backends stay comparable.
        q, clipped = _clip_skew(q, clipped_norm)
        r = torch.linalg.solve(eye + q, eye - q)
        return r, clipped

    # Neumann-based backends: clip is load-bearing for cayley_neumann.
    q, clipped = _clip_skew(q, clipped_norm)
    inv = _neumann_inverse(q, neumann_terms)
    r = inv @ (eye - q)
    if backend == "cans":
        r = _newton_schulz_orthogonalize(r, cans_iters)
    return r, clipped


class _OFTBase(nn.Module):
    """Shared OFT core: frozen base weight + learnable block-diagonal rotation.

    Subclasses provide the wrapped layer's flat 2-D weight ``[out, in]`` and a
    ``forward`` that consumes :meth:`effective_weight`. The base weight/bias live
    as **non-trainable buffers** (never optimised); only the per-block generator
    ``A`` (and, under DoRA-OFT, a magnitude vector) is learned. ``A`` is zero-init
    => ``R = I`` => the wrapped layer is an exact no-op at start.
    """

    def __init__(
        self,
        weight2d: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        blocks: int = 8,
        backend: str = "cayley_neumann",
        clipped_norm: float | None = 0.95,
        neumann_terms: int = 6,
        cans_iters: int = 3,
        dora: bool = False,
    ) -> None:
        super().__init__()
        out_features, in_features = weight2d.shape
        n_blocks = _largest_divisor_at_most(out_features, blocks)
        bs = out_features // n_blocks

        self.out_features = out_features
        self.in_features = in_features
        self.n_blocks = n_blocks
        self.block_size = bs
        self.backend = backend
        self.clipped_norm = clipped_norm
        self.neumann_terms = neumann_terms
        self.cans_iters = cans_iters
        self.dora = dora

        self.register_buffer("base_weight", weight2d.detach().clone().float())
        if bias is not None:
            self.register_buffer("base_bias", bias.detach().clone().float())
        else:
            self.base_bias = None

        self.oft_a = nn.Parameter(torch.zeros(n_blocks, bs, bs))
        if dora:
            self.oft_m = nn.Parameter(weight2d.detach().float().norm(dim=1).clone())
        else:
            self.oft_m = None

    def effective_weight(self) -> torch.Tensor:
        """Merged weight ``W' = R @ W`` as a flat ``[out, in]`` tensor."""
        r, clipped = skew_to_rotation(
            self.oft_a,
            backend=self.backend,
            clipped_norm=self.clipped_norm,
            neumann_terms=self.neumann_terms,
            cans_iters=self.cans_iters,
        )
        if clipped and self.training:
            _note_clip()
        w = self.base_weight.view(self.n_blocks, self.block_size, self.in_features)
        w = (r @ w.to(r.dtype)).reshape(self.out_features, self.in_features)
        if self.oft_m is not None:
            w = self.oft_m.view(-1, 1) * F.normalize(w, dim=1)
        return w.to(self.base_weight.dtype)


class OFTLinear(_OFTBase):
    """OFT wrapper for an ``nn.Linear`` (or any 2-D weight layer)."""

    def __init__(self, base: nn.Linear, **kw) -> None:
        super().__init__(base.weight, base.bias, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.effective_weight().to(x.dtype)
        bias = self.base_bias.to(x.dtype) if self.base_bias is not None else None
        return F.linear(x, weight, bias)

    def to_merged_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.base_bias is not None)
        lin = lin.to(self.base_weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(self.effective_weight())
            if self.base_bias is not None:
                lin.bias.copy_(self.base_bias)
        return lin


class OFTConv2d(_OFTBase):
    """OFT wrapper for a **pointwise (1×1, groups=1)** ``nn.Conv2d``.

    The 1×1 conv weight ``[out, in, 1, 1]`` squeezes to a ``[out, in]`` channel
    mixer, so the identical rotation applies; the merged layer is a plain 1×1
    ``nn.Conv2d`` indistinguishable from the original.
    """

    def __init__(self, base: nn.Conv2d, **kw) -> None:
        w2d = base.weight.detach().reshape(base.out_channels, base.in_channels)
        super().__init__(w2d, base.bias, **kw)
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.effective_weight().to(x.dtype).view(self.out_features, self.in_features, 1, 1)
        bias = self.base_bias.to(x.dtype) if self.base_bias is not None else None
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation)

    def to_merged_conv(self) -> nn.Conv2d:
        conv = nn.Conv2d(
            self.in_features, self.out_features, kernel_size=1,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            bias=self.base_bias is not None,
        ).to(self.base_weight.dtype)
        with torch.no_grad():
            conv.weight.copy_(self.effective_weight().view(self.out_features, self.in_features, 1, 1))
            if self.base_bias is not None:
                conv.bias.copy_(self.base_bias)
        return conv


# Throttled "norm was clipped" logging — divergence guards firing every batch
# would flood the log, so we report at most once per process by default.
_clip_logged = False


def _note_clip() -> None:
    global _clip_logged
    if not _clip_logged:
        logger.info(
            "OFT clipped-norm guard engaged (Neumann skew norm exceeded threshold "
            "and was scaled back; further occurrences suppressed)."
        )
        _clip_logged = True


def _is_pointwise_conv(m: nn.Module) -> bool:
    return (
        isinstance(m, nn.Conv2d)
        and m.kernel_size == (1, 1)
        and m.groups == 1
    )


def _named_target_modules(model: nn.Module):
    """Yield ``(parent, attr, module)`` for every wrappable backbone layer.

    Targets ``nn.Linear`` and pointwise (1×1, groups=1) ``nn.Conv2d`` — the
    channel-mixing layers — and skips the classifier head, depthwise/strided
    convs, and the stem.
    """
    for name, module in list(model.named_modules()):
        if name.startswith(_HEAD_PREFIX):
            continue
        for attr, child in list(module.named_children()):
            full = f"{name}.{attr}" if name else attr
            if full.startswith(_HEAD_PREFIX):
                continue
            if isinstance(child, nn.Linear) or _is_pointwise_conv(child):
                yield module, attr, child


def wrap_backbone_with_oft(
    model: nn.Module,
    *,
    blocks: int = 8,
    backend: str = "cayley_neumann",
    clipped_norm: float | None = 0.95,
    neumann_terms: int = 6,
    cans_iters: int = 3,
    dora: bool = False,
) -> int:
    """Replace every wrappable backbone layer with an OFT wrapper, in place.

    Freezes all non-OFT, non-head parameters (the base backbone) and leaves the
    head trainable. Returns the number of layers wrapped.
    """
    kw = dict(
        blocks=blocks, backend=backend, clipped_norm=clipped_norm,
        neumann_terms=neumann_terms, cans_iters=cans_iters, dora=dora,
    )
    targets = list(_named_target_modules(model))
    for parent, attr, child in targets:
        oft = OFTConv2d(child, **kw) if isinstance(child, nn.Conv2d) else OFTLinear(child, **kw)
        setattr(parent, attr, oft)

    # Freeze the frozen base: everything that is not an OFT generator and not in
    # the head. OFT params (oft_a / oft_m) keep requires_grad=True by default.
    for name, param in model.named_parameters():
        if ".oft_a" in name or ".oft_m" in name:
            param.requires_grad = True
        elif name.startswith(_HEAD_PREFIX):
            param.requires_grad = True
        else:
            param.requires_grad = False
    logger.info("Wrapped %d backbone Linear layers with OFT (backend=%s)", len(targets), backend)
    return len(targets)


def oft_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Trainable OFT generators + head tail (the params an OFT run optimises)."""
    params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if ".oft_a" in name or ".oft_m" in name or name.startswith(_HEAD_PREFIX):
            params.append(param)
    return params


def merge_oft_into_model(model: nn.Module) -> nn.Module:
    """Return a deep copy with every OFT wrapper collapsed to its base layer type.

    The result has the *vanilla* ConvNeXt module structure, so its
    ``state_dict()`` keys match a model that was never OFT-wrapped — exactly what
    ``load_checkpoint`` / ``backbone_feature_hash`` expect.
    """
    merged = copy.deepcopy(model)
    for parent, attr, child in [(p, a, c) for p, a, c in _iter_oft_modules(merged)]:
        replacement = (
            child.to_merged_conv() if isinstance(child, OFTConv2d) else child.to_merged_linear()
        )
        setattr(parent, attr, replacement)
    for param in merged.parameters():
        param.requires_grad = True
    return merged


def _iter_oft_modules(model: nn.Module):
    for name, module in list(model.named_modules()):
        for attr, child in list(module.named_children()):
            if isinstance(child, _OFTBase):
                yield module, attr, child


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Full-weight ``state_dict`` (vanilla keys) for an OFT-wrapped model."""
    return merge_oft_into_model(model).state_dict()
