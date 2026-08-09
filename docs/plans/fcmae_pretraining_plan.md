# FCMAE Self-Supervised Pretraining for BitTrainer — Implementation Plan

Status: APPROVED FOR IMPLEMENTATION. Repo: `F:\Projects\Other Tools\BItTrainer` (package
`bittrainer`). This plan is self-contained: every API named here has been verified against the
current source on 2026-07-24.

## Constraint block (non-negotiable)

- The repo has UNCOMMITTED work: never stash/reset/checkout/clean/revert; only additive changes;
  do not touch unrelated modified files.
- Do NOT commit or push.
- Never set CUDA_VISIBLE_DEVICES to "" — use "-1". Assume a training run may be using the GPU.
- Never kill any python process.
- Run ONLY the new test file plus directly-related existing tests (test_backbone_init.py); never
  the whole suite.
- New code follows house style: type hints, module docstrings explaining the why, logging via
  module logger, 100-char lines, double quotes (Ruff-formatted).

## How to run tests (verified working)

The clone's own `.venv` has NO pytest. Tests run with the Bitcrush Suite venv's python plus
`PYTHONPATH` pointed at the clone so `import bittrainer` resolves to the clone, not the pinned
site-packages copy:

```bash
cd "/f/Projects/Other Tools/BItTrainer" && \
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="F:/Projects/Other Tools/BItTrainer" \
"/f/Projects/Bitcrush/Bitcrush Suite/.venv/Scripts/python.exe" -m pytest \
bittrainer/tests/test_fcmae_pretraining.py -x -q
```

(Confirmed: this exact harness runs `bittrainer/tests/test_backbone_init.py` → 13 passed in 4.3s.
torch 2.12.1+cu130, timm 1.0.27, pytest 9.1.1.)

## Goal

A self-supervised FCMAE (fully convolutional masked autoencoder, ConvNeXt V2 paper) pretraining
path over a large pool of UNLABELED local images, producing a safetensors checkpoint with BARE
backbone keys that `bittrainer.backbone_init.apply_backbone_init` loads unchanged, with clean
provenance metadata (`pretrain_method="fcmae"`, `license_provenance="locally_trained"`,
`external_pretrained_used=false`). The run must survive days: pause/backup/resume, graceful
stop, stop-now — all via the existing `GenericTrainer` lifecycle.

## Files to create (ONLY these; no edits to existing files are required)

1. `F:\Projects\Other Tools\BItTrainer\bittrainer\fcmae_trainer.py` — masking/loss/decoder/
   dataset helpers + the async entry point `run_fcmae_pretraining`.
2. `F:\Projects\Other Tools\BItTrainer\bittrainer\generic\tasks\fcmae_task.py` — `FcmaeTask`,
   the `TrainingTask` subclass.
3. `F:\Projects\Other Tools\BItTrainer\bittrainer\tests\test_fcmae_pretraining.py` — TDD tests
   (write these FIRST).

Do NOT edit `backbone_init.py`, `model.py`, `generic/*.py` or any other existing file. If reuse
of a private helper is needed, import it (`import bittrainer.backbone_trainer as bb` — this is
established house precedent: `generic/tasks/backbone_task.py` already does exactly that).

## Verified API facts you will build against

- `bittrainer.model.create_model(model_size=..., pretrained=False, num_classes=0)` returns a
  timm ConvNeXt V2 with attributes `stem` (stride-4 Sequential), `stages` (4 × `ConvNeXtStage`;
  stage 0 downsample is Identity, stages 1–3 downsample ×2 → feature strides 4/8/16/32),
  `norm_pre` (Identity), `num_features` (e.g. atto=320). Verified by direct instantiation.
- `timm.models.convnext.ConvNeXtBlock(in_chs, out_chs=..., use_grn=True, ...)` is importable and
  is the V2 block when `use_grn=True`.
- `apply_backbone_init(module, spec)` (bittrainer/backbone_init.py) loads a safetensors file; if
  ALL keys start with `backbone.` it strips the prefix; then loads the intersection of matching
  key+shape tensors with `strict=False`; raises if the intersection is empty. So the export must
  use BARE timm keys (`stem.*`, `stages.*`, `norm_pre.*`, ...) and must NOT include any tensor
  namespaced `backbone.*`. Extra non-matching keys (e.g. `decoder.*`) would be tolerated by the
  matcher, but we exclude them anyway (decision D6).
- `GenericTrainer.run(task, progress_callback=..., stop_event=..., stop_now_event=...,
  pause_event=...)` (bittrainer/generic/generic_trainer.py) calls hooks in the order documented
  in `bittrainer/generic/task.py`. Key details:
  - `loop_spec()` is called BEFORE `prepare_data`, so anything `loop_spec` needs (e.g. whether a
    val split exists) must be computed in `make_context` (BackboneTask precedent).
  - The core steps `scheduler.step()` ONCE PER EPOCH and serialises `scheduler.state_dict()`
    into every backup. Do not step the scheduler inside `train_epoch`.
  - `boundary_hook(num_batches)` must be called at every gradient-accumulation boundary; it owns
    `global_step` and the pause/periodic-backup cadence; returning `"stop"` means break out of
    the epoch loop immediately (a pause backup was just written).
  - `improved = selected_score > best.best_validation_score + spec.selection_min_delta`; the
    core maximises, so a loss-based score must be negated.
  - Backups: `collect_epoch_state(...)` puts the FULL `model.state_dict()` (our wrapper,
    encoder+decoder), `optimizer.state_dict()`, `scheduler.state_dict()` in the envelope;
    task-specific extras ride `collect_extra_state`.
  - On successful completion the coordinator deletes all backups.
- `training_state.make_fingerprint(class_names=, num_classes=, max_epochs=, multi_label=,
  ordinal=, best_model_name=, model_size=)` returns a dict; extra keys may be added afterwards
  (BackboneTask adds `fingerprint["optimizer"]` and `fingerprint["resolution"]`).
  `fingerprint_matches(saved, current)` requires every key of `current` to match `saved`.
- `training_state.restore_optimizer_state(resume_state, optimizer, scheduler, device)` restores
  optimizer + scheduler; its Prodigy priming is a no-op for AdamW (it checks for `d_numerator`
  in param groups first). Safe to call unconditionally.
- `training_state.BackupCoordinator(backup_dir=..., backup_every_steps=..., pause_event=...,
  cb=...)`; `coordinator.load_resume(fingerprint, resume_from=...)` emits `resume_skipped` cb on
  fingerprint mismatch and returns None.
- `bittrainer.backbone_trainer` exports (module-private but house-reused): `_amp_settings(config)
  -> (enabled, dtype)`, `_stringify(value) -> str`, `_run_worker_async(worker, request,
  progress_callback, pause_event, stop_event=None, stop_now_event=None)` (the asyncio worker-
  thread + queue pattern), `_NORMALIZE` (ImageNet mean/std).
- `smart_cache._noop_callback` exists (BackboneTask imports it in `make_context`).
- `xxhash` is a project dependency (used by `model.backbone_feature_hash`).

## Design decisions (the "why")

### D1 — Dense-masked FCMAE approximation

The official FCMAE uses sparse convolutions so the encoder literally never touches masked
regions. timm models are dense, so we approximate:

- **Mask geometry**: one random mask per sample on the final-stage grid, i.e. patch size = 32
  input pixels (`image_size % 32 == 0` enforced with a clear `ValueError`). Grid `L = (H/32) *
  (W/32)`. Exact masked count `n_mask = int(round(mask_ratio * L))` chosen via `torch.randperm`
  (per-sample), so the masked fraction is exact, not Bernoulli.
- **Application**: the input image is multiplied by the visible mask upsampled ×32 (masked
  patches zeroed), AND the feature map is re-multiplied by the mask (upsampled to the feature
  stride via `repeat_interleave`) after the stem and after EVERY stage. All multiplications are
  out-of-place (`x = x * m`) — never in-place, so autograd stays intact.
- **Honesty about leakage**: within a single stage, dense 7×7 depthwise convs DO smear
  information across patch boundaries (a masked patch's features are influenced by visible
  neighbours before we re-zero them, and visible-patch features near a boundary see zeros where
  the official sparse encoder would see nothing at all). Re-masking after each stage bounds the
  leak to intra-stage receptive fields and prevents masked-position features from carrying
  content into the decoder — the decoder sees only the mask token at masked positions. This is
  the accepted dense approximation (equivalent to the ConvNeXt V2 paper's own note that FCMAE
  can be implemented densely by zeroing; the sparse path is a compute optimisation). The
  residual boundary leakage slightly eases the task; it does not collapse it, because the loss
  is computed only on masked patches whose features were zeroed at every stage boundary.
- **GRN**: already inside timm's V2 blocks — nothing to do.

### D2 — Decoder (lightweight, per the paper)

`_FcmaeDecoder(nn.Module)` in `fcmae_trainer.py`:

```python
class _FcmaeDecoder(nn.Module):
    def __init__(self, in_features: int, decoder_dim: int = 512, patch_size: int = 32) -> None:
        # proj: 1x1 conv in_features -> decoder_dim
        # mask_token: nn.Parameter(torch.zeros(1, decoder_dim, 1, 1))
        # block: timm ConvNeXtBlock(decoder_dim, use_grn=True)  (single block, per paper)
        # pred: 1x1 conv decoder_dim -> patch_size * patch_size * 3
    def forward(self, feat: torch.Tensor, grid_mask: torch.Tensor) -> torch.Tensor:
        # x = proj(feat); x = x * grid_mask + mask_token * (1 - grid_mask)  (out-of-place)
        # x = block(x); return pred(x)   # (B, p*p*3, gh, gw)
```

`grid_mask` is `(B, 1, gh, gw)` float, 1=visible, 0=masked. Mask token is trunc-normal-init
(std 0.02). Decoder dim 512 default (paper), configurable via `training_config["decoder_dim"]`
(tests use 64 to stay tiny).

### D3 — Loss: normalized-pixel MSE on masked patches only

`fcmae_loss(pred, target_images, bool_mask, *, patch_size=32, norm_pix=True)` in
`fcmae_trainer.py`:

- `_patchify(imgs, patch_size) -> (B, L, p*p*3)` via reshape/permute (MAE-style), where target
  images are the ORIGINAL (un-zeroed, ImageNet-normalized) inputs.
- When `norm_pix`: per-patch normalise the target: `(t - t.mean(-1, keepdim)) /
  sqrt(t.var(-1, keepdim) + 1e-6)`.
- `loss = ((pred - target) ** 2).mean(dim=-1)` → `(B, L)`; final scalar =
  `(loss * mask).sum() / mask.sum()` where `mask` is 1 on MASKED patches. Visible patches
  contribute exactly zero. Compute in fp32 (`pred.float()`, targets fp32) even under autocast.

### D4 — Optimizer/schedule: AdamW + linear warmup + cosine (NOT Prodigy_adv)

Evaluated both:

- *Prodigy_adv* (house factory, `generic/optimizer.py`): D-adaptation removes LR tuning and is
  validated in-house — but only on supervised objectives with the cosine schedule scaling its
  adapted step. MAE-family reconstruction losses have a very different curvature profile
  (heavy early transient while the decoder/mask-token learn); D-adaptation's global `d` estimate
  is driven by exactly that transient, and there is no in-house or published evidence for
  Prodigy on multi-day MAE pretraining. A mis-adapted `d` discovered three days into a run is an
  expensive failure mode.
- *AdamW + warmup + cosine* is the MAE/FCMAE convention (paper: AdamW, base_lr 1.5e-4 ×
  eff_bs/256, betas (0.9, 0.95), wd 0.05, warmup then cosine) with years of replication.

**Decision: AdamW.** Predictability beats adaptivity for a long unattended run, and the paper's
recipe transfers directly. Concretely:

- `lr = float(cfg.learning_rate or 1.5e-4) * eff_batch / 256` where
  `eff_batch = batch_size * accumulation_steps` (`learning_rate` is the BASE lr per 256).
- betas `(0.9, 0.95)`, `weight_decay = float(cfg.get("weight_decay", 0.05))`.
- Two param groups: params with `ndim <= 1` or name ending in `mask_token` get wd 0.0 (standard
  MAE exclusion of norms/biases/token).
- Scheduler: single `LambdaLR` stepped ONCE PER EPOCH by the core (do not step inside
  `train_epoch`): `mult(e) = (e + 1) / max(1, W)` for `e < W`, else
  `0.5 * (1 + cos(pi * (e - W) / max(1, T - W)))`, with `W = warmup_epochs` (default
  `max(1, round(0.05 * epochs))`), `T = epochs`. Epoch-granular LR is a deliberate
  approximation: with chunked "epochs" (D5) a run has tens-to-hundreds of scheduler steps, so
  the staircase is fine, and it keeps the core's scheduler serialisation/resume exactly as-is.
- Gradient accumulation: `accumulation_steps` (default 1). In `train_epoch`, scale
  `loss / accumulation_steps`, `backward()` each micro-batch, `optimizer.step()` +
  `zero_grad(set_to_none=True)` + `self.step += 1` + `boundary_hook(batches_done)` every
  `accumulation_steps` micro-batches; flush a trailing partial group at epoch end the same way.
- AMP: `bb._amp_settings(config)` (bf16 default, no GradScaler — house style); autocast wraps
  the model forward; loss in fp32 per D3.
- Fingerprint carries `"optimizer": "AdamW"` so any hypothetical stale envelope mismatches.

### D5 — "Epochs" = steps-per-epoch chunks; epoch-restart resume

A pass over a multi-million-image pool is not a useful backup/validation unit. So:

- `training_config["epochs"]` = number of CHUNKS (scheduler periods), and
  `training_config["steps_per_epoch"]` (optimizer steps; default 0 = one full pass over the
  train list). When `steps_per_epoch > 0`, each epoch consumes the first
  `steps_per_epoch * batch_size * accumulation_steps` entries of the freshly shuffled train
  list.
- `reshuffle()` shuffles `self.train_paths` with `random.shuffle` — python `random` is captured
  in the core's RNG snapshots, so a resumed run rebuilds the identical epoch layout.
- Resume is EPOCH-RESTART (BackboneTask precedent): `build_loaders` ignores `resume_info`,
  returns `(loader, None, 0)`. Mid-epoch periodic/pause backups still happen (model/optimizer
  state is preserved); a resume restarts the interrupted chunk from its top. With the
  recommended chunk sizes (~20–60 min) the worst-case loss is one chunk of compute — an
  acceptable trade for not carrying the schedule-replay machinery.
- Long-run cadence: `backup_every_steps` (default 0 = boundary/pause/exception only; recommend
  500–1000 for real runs) + one backup at every epoch boundary (core does this automatically
  when the coordinator is enabled).

### D6 — Export contract: bare trunk only, decoder excluded

`finalize` writes safetensors at `request["candidate_checkpoint_path"]` containing ONLY the
encoder (timm model) state dict with BARE keys — no `decoder.*`, no `backbone.` prefix — so
`apply_backbone_init` consumes it unchanged (its bare-key branch). The decoder is deliberately
NOT exported: the supervised fine-tune never consumes it, and "continue pretraining later" is
served by `backup_dir`/`resume_from` (backups carry the full wrapper incl. decoder + optimizer).
This keeps one artifact with one meaning.

String metadata (all values through `bb._stringify`, `None`s dropped — BackboneTask pattern):

- `pretrain_method="fcmae"`, `mask_ratio`, `patch_size` ("32"), `image_size`,
  `steps_completed`, `epochs_completed`, `dataset_image_count`, `dataset_fingerprint`
  (xxhash64 of the sorted path list — same value folded into the run fingerprint),
  `val_loss` (best held-out reconstruction loss; omit if no val split),
- `license_provenance` (request value or `"locally_trained"`),
  `external_pretrained_used` (bool from request, default false),
  `release_blocking` (bool from request),
- passthroughs when present: `family_name`, `architecture`, `size_alias`, `display_size`,
  `convnextv2_size`, `training_run_id` (= `request["run_id"]`),
- `version="1"`, `status="candidate"`, `created_at` (UTC ISO-Z, BackboneTask pattern),
  `training_config_json` (the config dict).

### D7 — Selection/monitoring without labels

- Held-out slice: deterministic split by `xxhash.xxh64(path_or_content_hash).intdigest() %
  10_000 / 10_000 < val_fraction` (NEVER python `hash()` — it is salted per process). When the
  request supplies `content_hashes` (path → hash), the hash is the split key so re-runs with
  moved files keep the same partition; else the absolute path string. `val_fraction` default
  0.02, and the val list is truncated to `val_max_images` (default 2000, deterministic: sort by
  the same digest and take the first N).
- Validation = mean fcmae loss over the val loader with a DETERMINISTIC PER-IMAGE mask: seed a
  `torch.Generator()` (CPU) with `xxhash.xxh64(path).intdigest() & 0x7FFFFFFF` per sample so
  every epoch scores the same masks — epoch-to-epoch val losses are comparable.
- `selection_score(metrics) = -metrics["val_loss"]` (core maximises); `selection_min_delta=0.0`.
- `save_candidate` snapshots `{k: v.detach().cpu().clone() ...}` of the ENCODER state dict
  (`.clone()` is load-bearing on CPU — BackboneTask comment explains aliasing).
- No val split (`val_fraction=0` or an empty draw): patience is set unreachable
  (`epochs + 1`), `validate` returns `{}`, `selection_score` returns 0.0, and `finalize`
  exports the FINAL encoder weights (BackboneTask precedent).
- No EMA. MAE-family pretraining does not use weight EMA (the paper fine-tunes the raw
  pretrained weights); adding house `ModelEMA` would double encoder memory for no evidenced
  gain. (Deliberate deviation from the supervised tasks.)
- Progress frames: identical shape to BackboneTask —
  `{"type": "training_progress", "stage": <"preparing"|"training"|"validating"|"saving">,
  "status_text": ..., "run_id": ..., "seq": <monotonic int>}` plus payload extras (train loss,
  val_loss, steps, lr, epoch/epochs). Per-epoch message via the `epoch_message` hook.

### D8 — Fingerprint (resume-compatibility identity)

`_fcmae_fingerprint(model_size, epochs, *, mask_ratio, image_size, dataset_fingerprint,
dataset_count, steps_per_epoch)` in `fcmae_trainer.py`:

```python
fp = make_fingerprint(
    class_names=[], num_classes=0, max_epochs=int(epochs), multi_label=False,
    ordinal=False, best_model_name="fcmae_candidate", model_size=str(model_size),
)
fp["trainer"] = "fcmae"
fp["optimizer"] = "AdamW"
fp["mask_ratio"] = float(mask_ratio)
fp["resolution"] = str(image_size)
fp["dataset"] = f"{dataset_fingerprint}:{dataset_count}"
fp["steps_per_epoch"] = int(steps_per_epoch)
return fp
```

Any change to mask ratio, resolution, model size, dataset membership, chunking or epoch budget
cleanly mismatches a stale backup (`resume_skipped` → fresh start) instead of loading
incompatible state.

### D9 — Data pipeline

In `fcmae_trainer.py`:

- Request accepts `images: list[str]` and/or `image_roots: list[str]`. Roots are walked
  recursively (`Path.rglob("*")`) for suffixes `{".jpg", ".jpeg", ".png", ".webp", ".bmp"}`
  (case-insensitive); skip any filename containing `-masklabel` (house dataset hygiene). The
  union is deduped: by `content_hashes[path]` when provided, else by resolved absolute path.
  Sorted for determinism BEFORE hashing/splitting.
- `_FcmaeDataset(paths, transform)`: `__getitem__` opens with PIL, `convert("RGB")`, transform.
  Unreadable/corrupt files: catch `(OSError, ValueError, Image.DecompressionBombError,
  SyntaxError)`, log a warning (once per path — keep a module-level or instance `set`), return
  `None`. `_collate_drop_none(batch)` filters `None`s and returns a stacked tensor, or `None`
  when the whole batch failed; `train_epoch`/validate skip `None`/empty batches.
- Train transform (MAE practice — augmentation-light):
  `RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC)` +
  `RandomHorizontalFlip(0.5)` + `ToTensor` + `bb._NORMALIZE`. No color jitter.
- Val transform: `Resize((image_size, image_size))` + `ToTensor` + `bb._NORMALIZE`
  (deterministic).
- `num_workers=0` in tests; honor `training_config["dataloader_workers"]` via
  `training_state.loader_kwargs(n)` for real runs (default 4).

### D10 — Wrapper module and forward

In `fcmae_trainer.py`:

```python
class _FcmaeModel(nn.Module):
    """Encoder (timm ConvNeXt V2, num_classes=0) + FCMAE decoder as one module.

    One module so the backup envelope carries a single state_dict (house pattern:
    _BackboneWithHeads). ``encoder`` holds BARE timm keys under the ``encoder.`` namespace;
    export strips it.
    """
    def __init__(self, encoder: nn.Module, decoder: _FcmaeDecoder, patch_size: int = 32): ...

    def forward_masked(self, images: torch.Tensor, grid_mask: torch.Tensor) -> torch.Tensor:
        # m32 = upsample(grid_mask, 32); x = images * m32
        # x = encoder.stem(x);  x = x * upsample(grid_mask, stem_out_scale)
        # for stage in encoder.stages: x = stage(x); x = x * upsample_to(x, grid_mask)
        # (compute the per-stage upsample factor from x.shape[-1] // grid_mask.shape[-1] —
        #  robust to any stage geometry; all multiplies out-of-place)
        # return decoder(x, grid_mask)   # (B, p*p*3, gh, gw)
```

Helpers (module-level, unit-testable):

- `make_random_masks(batch: int, grid_h: int, grid_w: int, mask_ratio: float,
  generator: torch.Generator | None = None) -> torch.Tensor` — returns `(B, 1, gh, gw)` float
  visible-mask (1=visible) with EXACTLY `int(round(mask_ratio * L))` zeros per sample
  (`torch.randperm` per sample, honoring `generator` when given).
- `_upsample_mask(grid_mask, scale) -> Tensor` — `repeat_interleave(scale, -2)
  .repeat_interleave(scale, -1)`.
- `_patchify(images, patch_size) -> (B, L, p*p*3)`.
- `fcmae_loss(pred, images, grid_mask, *, norm_pix=True) -> torch.Tensor` — pred is the
  decoder output `(B, p*p*3, gh, gw)`; flatten to `(B, L, p*p*3)`; masked-only mean per D3
  (`masked = 1 - grid_mask` flattened to `(B, L)`).

### D11 — Async entry point

In `fcmae_trainer.py` (mirrors `run_backbone_training` exactly):

```python
async def run_fcmae_pretraining(
    request: dict,
    progress_callback=None,
    *,
    pause_event=None,
    stop_event=None,
    stop_now_event=None,
) -> dict:
    return await bb._run_worker_async(
        _train_fcmae, request, progress_callback, pause_event, stop_event, stop_now_event
    )


def _train_fcmae(request, emit, stop, pause_event=None, stop_event=None, stop_now_event=None):
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.fcmae_task import FcmaeTask

    task = FcmaeTask(request, cancel_event=stop, stop_event=stop_event)
    return GenericTrainer().run(
        task, progress_callback=emit, pause_event=pause_event,
        stop_event=task.steps_stop_event, stop_now_event=stop_now_event,
    )
```

Plus `class FcmaeTrainingCancelled(RuntimeError)` raised from `train_epoch` when
`cancel_event.is_set()` (BackboneTask pattern: cancel ≠ graceful stop). `max_steps` (optional)
rides `steps_stop_event` exactly like BackboneTask (`<= 0` → set before epoch 0; reached →
`steps_stop_event.set()` + break; the event doubles as the caller's finish-early stop_event).

### D12 — Request contract (document in the module docstring)

```python
{
    "run_id": str,
    "family_name"/"architecture"/"size_alias"/"display_size"/"convnextv2_size": str,  # optional
    "candidate_checkpoint_path": str,           # REQUIRED: where the safetensors goes
    "images": [str, ...],                       # explicit files (optional)
    "image_roots": [str, ...],                  # folders walked recursively (optional)
                                                # at least one of the two must yield images
    "content_hashes": {path: hash},             # optional; dedup + stable split keys
    "training_config": {
        "image_size": 224,          # must be a multiple of 32
        "batch_size": 32,
        "accumulation_steps": 1,
        "epochs": 20,               # CHUNKS (scheduler periods)
        "steps_per_epoch": 0,       # optimizer steps per chunk; 0 = one full pass
        "max_steps": None,          # optional global optimizer-step cap
        "mask_ratio": 0.6,
        "decoder_dim": 512,
        "norm_pix_loss": True,
        "learning_rate": 1.5e-4,    # BASE lr per eff-batch 256 (scaled internally)
        "weight_decay": 0.05,
        "warmup_epochs": None,      # default max(1, round(0.05 * epochs))
        "validation_split": 0.02,   # val_fraction; 0 disables the held-out slice
        "val_max_images": 2000,
        "device": None,             # "cpu"/"cuda"; default auto
        "use_amp": True, "amp_dtype": "bfloat16",
        "dataloader_workers": 4,
        "backup_dir": None, "backup_every_steps": 0, "resume_from": None,
    },
    "license_provenance": "locally_trained",
    "external_pretrained_used": False,
    "release_blocking": False,
}
```

Result dict from `finalize`:
`{"candidate_checkpoint_path", "val_loss" (float | None), "steps_completed",
"epochs_completed", "dataset_image_count", "best_epoch"}`.

## FcmaeTask hook-by-hook spec (`bittrainer/generic/tasks/fcmae_task.py`)

`class FcmaeTask(TrainingTask)`, `trainer_name = "fcmae"`, reaching helpers via
`import bittrainer.fcmae_trainer as ft` (module-alias pattern per house docs). Constructor
`__init__(self, request: dict, *, cancel_event=None, stop_event=None)` parses the config like
`BackboneTask.__init__` (incl. `steps_stop_event` logic and `max_steps <= 0`).

- `make_context`: resolve device (config override else cuda-if-available); enumerate + dedup +
  sort the image paths (`ft._gather_images(request)`); raise `RuntimeError` when empty; compute
  `self.dataset_fingerprint` (xxhash64 hex of `"\n".join(sorted_keys)` where key = content hash
  or path) and split train/val (`ft._split_paths(paths, hashes, val_fraction, val_max)`);
  encoder is NOT created here. Emitter = the no-op `.stage` `_Emitter` (copy the BackboneTask
  inner class). `checkpoint_dir = Path(request["candidate_checkpoint_path"]).parent`. Model is
  always created with `pretrained=False` (`create_model(model_size=..., pretrained=False,
  num_classes=0)`) — NEVER timm pretrained: the whole point is clean provenance.
- `fingerprint_init`: `BackupCoordinator(backup_dir=..., backup_every_steps=...,
  pause_event=ctx.pause_event, cb=ctx.cb)`; fingerprint via `ft._fcmae_fingerprint(...)` (D8);
  `load_resume` only when `resume_from` set. (BackboneTask verbatim shape.)
- `loop_spec`: `LoopSpec(max_epochs=self.epochs, patience=self.patience if (self.patience > 0
  and self.val_paths) else self.epochs + 1, selection_min_delta=0.0)`.
- `prepare_data`: emit `"preparing"` frame with `train_images`, `val_images`,
  `dataset_fingerprint`, `image_size`, `mask_ratio`.
- `create_model`: build encoder (`pretrained=False`, `num_classes=0`) + `ft._FcmaeDecoder(
  encoder.num_features, decoder_dim)`, wrap in `ft._FcmaeModel`, `.to(ctx.device)`; on resume,
  `model.load_state_dict(resume_state["model"])`. Store `self.model`.
- `resolve_batch_size`: return configured `batch_size` (no autobatch).
- `create_optimizer`: AdamW per D4 (param-group split helper `ft._param_groups(model,
  weight_decay)`); `LambdaLR` warmup+cosine per D4; `t_max = max(1, self.epochs)`; on resume
  `restore_optimizer_state(...)` + `self.step = int(resume_state.get("global_step", 0))`.
  Return `(optimizer, scheduler, t_max)`.
- `restore_resume_extra`: reload `best_encoder_state` + best val loss from the envelope
  (mirror BackboneTask's `restore_resume_extra`).
- `resumed_message`: `{"type": "training_resumed", "run_id", "resumed_from", "epoch",
  "global_step", "best_score"}`.
- `reshuffle`: `random.shuffle(self.train_paths)`.
- `build_loaders`: build the val loader ONCE (lazy, deterministic order `shuffle=False`);
  epoch list = `self.train_paths[:steps_per_epoch * batch_size * accumulation_steps]` when
  `steps_per_epoch > 0` else the whole list; train `DataLoader(ft._FcmaeDataset(epoch_paths,
  ft._train_transform(image_size)), batch_size=..., shuffle=True,
  collate_fn=ft._collate_drop_none, **loader_kwargs(workers))`. Return `(loader, None, 0)`.
- `train_epoch`: per D4/D11 — cancel check → max_steps check → skip `None` batches → autocast
  forward: `grid_mask = ft.make_random_masks(b, gh, gw, self.mask_ratio)` on device →
  `pred = model.forward_masked(images, grid_mask)` → fp32 `loss = ft.fcmae_loss(...)` →
  `(loss / accum).backward()` → step/zero/`self.step += 1`/`boundary_hook` at accumulation
  boundaries (break on `"stop"`); flush trailing partial group. Return mean loss.
- `validate`: no val → `{}`. Else `torch.no_grad()`, autocast forward with the per-image
  deterministic masks (D7), mean loss → `{"val_loss": float}`. Also emit nothing here (the core
  emits the validating stage via `ctx.em.stage`, which our emitter no-ops — match BackboneTask
  by keeping validation quiet until `epoch_message`).
- `selection_score`: `-metrics["val_loss"]` if metrics else `0.0`.
- `save_candidate`: no-op without val; else clone the ENCODER state to
  `self.best_encoder_state` (`.detach().cpu().clone()` — aliasing note), store
  `self.best_val_loss`.
- `epoch_message`: `_emit("training", f"Epoch {epoch+1}/{self.epochs}", epoch=..., epochs=...,
  steps=self.step, loss=train_result, val_loss=..., lr=optimizer? — lr is not passed to this
  hook; read `self._last_lr` captured in train_epoch from
  `optimizer.param_groups[0]["lr"]`, best_val_loss=...)`; return None.
- `collect_extra_state`: `{"rng_now": capture_rng_states(ctx.device), "rng_epoch_start":
  rng_epoch_start, "best_encoder_state": self.best_encoder_state, "best_val_loss":
  self.best_val_loss}`.
- `finalize`: emit `"saving"`; choose best encoder state (or final when no val/no improvement:
  `{k: v.detach().cpu() for ...}` of the live encoder); write safetensors with bare keys +
  metadata per D6; return the result dict per D12.

## Tests FIRST — `bittrainer/tests/test_fcmae_pretraining.py`

Style: CPU (`device: "cpu"` in config), atto, 64px images, `decoder_dim=64`, few steps,
`dataloader_workers=0`, module docstring explaining what is pinned, plain functions + tmp_path,
`asyncio.run(...)` for the entry point (copy `test_backbone_generic.py` scaffolding:
`_make_images`, `_request`, `_run`, `_FlagEvent`-style pause helpers from
`test_training_resume.py`). All tests must pass under `CUDA_VISIBLE_DEVICES=-1`.

1. `test_make_random_masks_exact_fraction` — for `(gh, gw, ratio)` in
   `[(2, 2, 0.6), (4, 4, 0.6), (4, 4, 0.75), (7, 5, 0.5)]`, batch 3: every sample has exactly
   `int(round(ratio * gh * gw))` zeros; values are only 0/1; shape `(B, 1, gh, gw)`.
2. `test_loss_ignores_visible_patches` — build a 2-image batch, fixed grid mask; construct
   `pred` that EQUALS the per-patch-normalized target on masked patches and is garbage
   (e.g. +1000) on visible patches → `fcmae_loss` ≈ 0 (`abs < 1e-6`). Then perturb ONLY the
   visible patches of a random pred → loss unchanged (`torch.isclose`). Proves the mask is on
   the loss, not just the input.
3. `test_forward_backward_smoke_cpu` — atto encoder (`pretrained=False`), 64px, batch 2,
   `mask_ratio=0.6`: forward_masked output shape `(2, 32*32*3, 2, 2)`; loss finite;
   `loss.backward()` puts a non-zero grad on `encoder.stem[0].weight` AND on the decoder
   mask_token (proves gradient flows through masking; no in-place autograd breakage).
4. `test_feature_masking_blocks_leakage` — with a fixed input, run `forward_masked`; assert
   the encoder features handed to the decoder are exactly zero at masked grid positions
   (hook the decoder or expose the pre-decoder features via a small refactor-friendly seam:
   simplest is to test `_upsample_mask` + verify `model.forward_masked` output changes when a
   VISIBLE patch pixel changes but is INVARIANT when a MASKED patch's input pixels change —
   the invariance is the honest anti-leak claim at the decoder input).
5. `test_end_to_end_pretrain_and_export` — 8 tiny images, `epochs=2`, `steps_per_epoch=2`,
   `batch_size=2`, `validation_split=0.25` with supplied `content_hashes`; run
   `run_fcmae_pretraining`; assert result has `candidate_checkpoint_path` +
   `steps_completed > 0`; `safe_open` the file: NO key starts with `decoder.` or `backbone.`;
   keys include `stem.0.weight`; metadata `pretrain_method == "fcmae"`,
   `mask_ratio == "0.6"`, `external_pretrained_used == "false"`,
   `license_provenance == "locally_trained"`, `dataset_image_count` correct.
6. `test_export_loads_via_apply_backbone_init` — take test 5's checkpoint;
   `create_model(model_size="atto", pretrained=False, num_classes=2)`;
   `apply_backbone_init(model, {"source": "local_candidate", "checkpoint_path": path})` is
   True; `stem.0.weight` equals the checkpoint tensor (supervised fine-tune consumption path).
7. `test_backup_resume_roundtrip` — run A with `backup_dir` + pause_event flipped by a
   callback after the first `epoch_complete`-equivalent frame (the epoch "training" message
   with `epoch == 1`) → returns `{"paused": True, ...}` and a backup exists; run B same
   request + `resume_from=backup_dir` → emits `training_resumed` with `epoch == 1`, completes
   `epochs_completed == 2`, and `steps_completed` strictly greater than at pause. Assert the
   resumed optimizer really restored: capture run B's optimizer state via monkeypatching
   `ft`-side? Keep it behavioural: assert the `training_resumed` frame's `global_step > 0`
   (optimizer/scheduler restore is exercised by `restore_optimizer_state`, and continuity is
   the observable).
8. `test_fingerprint_mismatch_starts_fresh` — pause run A as in test 7; run B with
   `mask_ratio=0.75` + `resume_from` → a `resume_skipped` frame with
   `reason == "fingerprint_mismatch"` is emitted and training starts at epoch 0.
9. `test_holdout_split_deterministic` — `_split_paths` twice on the same 200 fake paths →
   identical partitions; val fraction within [0.5×, 2×] of requested at that N; disjoint and
   covering; and truncation to `val_max_images` keeps the lexicographically-stable digest
   ordering (call it twice → same truncated set).
10. `test_unreadable_file_skipped` — include one text file renamed `.png` among 6 good images;
    end-to-end run completes; no exception; (optional) caplog shows one warning for the bad
    path.

Existing related test to also run (and NOT break): `bittrainer/tests/test_backbone_init.py`.

## Acceptance checklist for the implementer

- [ ] Tests written first, then implementation until green.
- [ ] `pytest bittrainer/tests/test_fcmae_pretraining.py -x -q` green via the harness command
      above; paste the real output.
- [ ] `pytest bittrainer/tests/test_backbone_init.py -q` still green.
- [ ] No existing file modified (git status shows only the three new files + this plan).
- [ ] No in-place tensor mutation in the masked forward/loss path.
- [ ] All emitted frames carry `type`/`stage`/`status_text`/`seq`.
- [ ] Module docstrings explain the dense-masking approximation and the AdamW decision.
