# Trainer Parity Round — Backbone Maturity + Shared-Factory Fixes + Engine Split Integrity

Design lead plan. Red tests are already written (see "Test inventory"); the implementation
agent turns them green WITHOUT modifying them. Scope: BitTrainer workstreams A + B, Engine
workstream C item 9 (pure logic only). FCMAE and soft-implicit-negative/ASL redesign are
OUT of scope.

## Constraint block (verbatim, binding)

- Both repos have UNCOMMITTED work: never stash/reset/checkout/clean/revert; additive
  changes only; do not touch files outside the planned list; NEVER touch fcmae-named files.
- Do NOT commit or push in either repo.
- Never set CUDA_VISIBLE_DEVICES to "" — use "-1". Assume a training run may own the GPU.
- Never kill any python process.
- Run only the tests named in the plan (new files + listed related existing files); never a
  whole suite.
- Do not modify the design lead's test files; report disputes instead.
- House style: Ruff-formatted, type hints, docstrings explaining the why, 100-char lines,
  double quotes.

## Files the implementation may touch (exhaustive)

BitTrainer (`F:\Projects\Other Tools\BItTrainer`):
- `bittrainer/backbone_trainer.py`
- `bittrainer/generic/tasks/backbone_task.py`
- `bittrainer/generic/tasks/backbone_heads_task.py` (only what B-items force; behaviour
  must stay green in `test_backbone_heads_task.py`)
- `bittrainer/generic/optimizer.py`
- `bittrainer/model.py`
- `bittrainer/group_trainer.py` (GroupTrainConfig fields + `_train_one_epoch` clip wiring +
  `_make_optimizer` pass-through + `_create_or_warmstart_model` drop-path pass-through)
- `bittrainer/generic/tasks/group_task.py` (fingerprint optimizer identity + resume-path
  create_model drop-path pass-through)
- `bittrainer/training_state.py` (ONLY the additive `optimizer_identity` kwarg on
  `make_fingerprint`)

Bitcrush Suite (`F:\Projects\Bitcrush\Bitcrush Suite`):
- `apps/engine/src/backend/backend/services/val_split.py` (additive functions only)

Test files (READ-ONLY for the implementer — authored by the design lead):
- `bittrainer/tests/test_backbone_selection_metrics.py`
- `bittrainer/tests/test_backbone_calibration_export.py`
- `bittrainer/tests/test_backbone_augment_parity.py`
- `bittrainer/tests/test_llrd_wrapper_prefixes.py`
- `bittrainer/tests/test_optimizer_wd_exclusions.py`
- `bittrainer/tests/test_drop_path_factory.py`
- `bittrainer/tests/test_grad_clipping.py`
- `apps/engine/tests/backend/test_val_split_neardup.py`
- (one pre-made surgical edit by the design lead: `bittrainer/tests/test_backbone_resolution.py`
  `_spy_transforms` now forwards `*a, **k` — already applied, keep it.)

## Workstream A — backbone builder maturity

### A1. Selection metric: per-head F1/AP instead of mean raw accuracy

Today `bb._evaluate` returns `{binary/<c>: acc@logit>0, group/<g>: acc}` and
`BackboneTask.selection_score` is the blind mean. Accuracy at logit>0 saturates on
imbalanced heads and the blind mean is dominated by easy heads.

Design (all in `bittrainer/backbone_trainer.py`):

- `_collect_val_scores(backbone, heads, loader, device, *, amp_enabled=False,
  amp_dtype=torch.bfloat16) -> tuple[dict, dict]` — one pass over the (masked-unknown) val
  loader collecting raw logits: `binary_scores[concept] = (logits[N] np.float64,
  targets[N] np.float64)`, `group_scores[group] = (logits[N,C], targets[N])`. Heads run in
  fp32 exactly like the current `_evaluate`.
- `_backbone_metrics(binary_scores, group_scores) -> dict[str, float]` — pure, unit-tested:
  - Legacy keys preserved byte-for-byte: `binary/<c>` (accuracy at logit>0), `group/<g>`
    (argmax accuracy) — additive wire contract.
  - New per-head keys: `binary_f1/<c>` (F1 at sigmoid>0.5 == logit>0), `binary_ap/<c>`
    (`sklearn.metrics.average_precision_score`), `binary_support/<c>` (positive count in
    val). F1/AP are only emitted for heads with >= 1 positive; support is always emitted.
  - New per-group keys: `group_f1/<g>` (macro-F1 over that group's classes via
    `group_validation.compute_multiclass_metrics`), `group_val_support/<g>` (row count).
  - Aggregates: `binary_macro_f1` = mean of `binary_f1/*` over heads with positive support
    (omitted when no head qualifies); `group_macro_f1` = mean of `group_f1/*` over groups
    with rows (omitted when none). `selection` = mean of whichever of the two aggregates
    exist, else 0.0. Binary and group are therefore reported separately AND combined in a
    defensible way: equal weight to the binary panel and the group panel, zero-support
    heads excluded rather than contributing frozen zeros.
- `bb._evaluate` keeps its name/signature (it is a live monkeypatch seam in
  `test_backbone_resolution.py`) and becomes: collect via `_collect_val_scores`, compute via
  `_backbone_metrics`. It gains an optional `score_sink: dict | None = None` kwarg; when
  provided, the raw `{"binary": binary_scores, "groups": group_scores}` are stashed there
  (calibration needs the logits of the winning epoch; fakes that ignore kwargs simply leave
  the sink empty).
- `BackboneTask.validate` passes `score_sink`; keeps `validation_metrics` = full dict and
  `validation_score` = `selection_score(metrics)`.
- `BackboneTask.selection_score(metrics)` returns `metrics["selection"]` when present, else
  the legacy mean of the values (keeps the `bb._evaluate` fakes in existing tests scoring
  0.9/0.1 exactly as before).
- `save_candidate` additionally snapshots the sink into `self._best_val_scores` so
  calibration is fitted on the exported epoch, not the last one.

Guards: heads with zero val positives are excluded from `binary_macro_f1` and appear only
as `binary_support/<c> = 0`; a group with zero val rows contributes nothing. AP is skipped
(never NaN) without positives.

### A2. Calibration export: per-head thresholds (+ optional temperature)

- `_tune_binary_thresholds(binary_scores, *, min_positive=1) -> dict[str, float]` in
  `backbone_trainer.py`: per head, reuse
  `group_validation.find_per_class_thresholds(probs[:, None], targets[:, None])` on
  `sigmoid(logits)` (one column per call — it already implements the F1-optimal grid sweep
  and the min-positive 0.5 fallback). Heads below `min_positive` positives keep 0.5.
- `_fit_binary_temperature(logits, targets) -> float`: deterministic grid search over
  `T in geomspace(0.25, 4.0, 33)` minimising mean BCE NLL of `sigmoid(logits / T)`.
- Config keys (training_config, additive): `calibrate_thresholds: bool = True`,
  `calibrate_temperature: bool = False` (off: temperature is a second-order win and adds a
  consumer-side contract; opt in per run).
- `BackboneTask.finalize`: when a val split exists and best scores were captured, fit and
  fold into the candidate metadata via `_candidate_metadata`:
  - `binary_thresholds_json`: `{concept: threshold}` (always when calibrate_thresholds and
    val scores exist),
  - `binary_temperatures_json`: `{concept: T}` (only when calibrate_temperature).
  Consumers keep working if the keys are absent (additive). Decision rule documented in
  the metadata consumer: `sigmoid(logit / T) >= threshold`, with T defaulting to 1.0 and
  threshold to 0.5.
- `BackboneHeadsTask` inherits `_candidate_metadata`; its finalize does not fit calibration
  this round (cached-vector val logits would be needed — follow-up), and must keep passing
  its existing tests. The metadata keys are simply absent there.

### A3. Augmentation parity (config-gated, val stays deterministic)

- `_train_transform(image_size, aug: dict | None = None)`:
  `aug=None` == defaults ON (parity with the group trainer, which runs RandAugment(2,9),
  RandomErasing 0.25 and hflip unconditionally — that IS the justified default):
  - `use_random_resized_crop: bool = True` -> `RandomResizedCrop(image_size,
    scale=(rrc_scale_min, 1.0))` with `rrc_scale_min: float = 0.6`; False -> legacy squash
    `Resize((s, s))`.
  - `randaugment_n: int = 2`, `randaugment_m: int = 9` -> `transforms.RandAugment` on PIL
    before the crop; `n == 0 or m == 0` disables.
  - `random_erasing_p: float = 0.25` -> `RandomErasing(p)` after Normalize; `0` disables.
  - Keep the existing HFlip + ColorJitter.
- `_val_transform(image_size)`: DECIDED — aspect-preserving `Resize(image_size)` (shorter
  side) + `CenterCrop(image_size)` (the ImageNet eval convention; matches what an
  RRC-trained model sees and removes the squash distortion). Deterministic by construction.
- Label smoothing for the group-CE heads: `_batch_loss(..., group_label_smoothing: float
  = 0.0)` forwarded to `F.cross_entropy(label_smoothing=...)`. Config
  `group_label_smoothing: float = 0.1` on the task (parity with the group default).
- Loss normalization (audit fold-in): `_batch_loss(..., reduction: str = "sum")` —
  `"mean"` averages over the contributing head losses so gradient scale no longer grows
  with per-batch label density. Task config `loss_reduction: str = "mean"` (new default
  for backbone runs; the function default stays `"sum"` so direct legacy callers are
  unchanged). Justification: head-count-invariant loss scale; Prodigy re-adapts `d`.
- Positive-cap selection (audit fold-in): `_plan_epoch_samples(...,
  positive_cap_mode: str = "density")` — `"random"` takes a plain seeded `rng.sample`
  instead of the density-first sort. Function/task default stays `"density"` (existing
  behaviour + tests preserved); `"random"` is the recommended A/B (density-first
  systematically over-selects multi-label images). Task reads `positive_cap_mode`.
- `BackboneTask` builds `self._aug` from config once and calls
  `bb._train_transform(size, self._aug)`. (The `_spy_transforms` forward-compat edit in
  `test_backbone_resolution.py` is already applied.)

### A4. LLRD for the backbone task

- `model.build_llrd_param_groups`: bucket on the name AFTER stripping one optional leading
  `"backbone."`; names starting `"heads."` (the multi-task wrapper) or `"head."` go to the
  `head` bucket (depth 0). Plain models bucket exactly as before.
- `make_optimizer(model, llrd=..., llrd_decay=...)` already threads through; `BackboneTask`
  config gains `llrd: bool = True`, `llrd_decay: float = 0.8` (parity with the group
  default `llrd=True`) and `create_optimizer` passes them (plus `wd_exclusions`, B7).
- Fingerprint: `generic/optimizer.py` gains
  `optimizer_identity(*, llrd: bool, llrd_decay: float, wd_exclusions: bool) -> str`
  returning exactly `"Prodigy_adv"` for (False, *, False) — the historical string — else
  e.g. `"Prodigy_adv+llrd0.8+nd"`. `bb._backbone_fingerprint` gains
  `optimizer_identity: str = "Prodigy_adv"` kwarg used for `fingerprint["optimizer"]`;
  `BackboneTask.fingerprint_init` passes the computed identity. A stale flat-AdamW/flat-
  Prodigy backup then cleanly mismatches (resume_skipped) instead of loading a
  different-group-count optimizer state.
- `BackboneHeadsTask` keeps `make_optimizer(model.heads)` (flat + legacy exclusions
  default) — heads-only retraining has one depth; no LLRD there.

### A5. Greedy soup + autobatch — decision

- **Autobatch: IN, opt-in.** `training_config.batch_size` of `0` or `"auto"` makes
  `resolve_batch_size` run `bittrainer.autobatch.determine_batch_size` with
  `bucket_counts={(image_size, image_size): len(train_samples)}` (backbone trains square)
  and emit the standard `{"type": "autobatch", ...}` frame. Engine's explicit batch_size
  keeps today's path bit-for-bit; the default `or 8` fallback is unchanged, so the wire
  contract is strictly additive.
- **Greedy soup: CUT this round.** Rationale: (a) the backbone keeps its best state
  in-memory, so a soup pool means a new on-disk candidate store + eviction machinery;
  (b) every soup evaluation is a full val pass at 384px over backbone+heads — the group
  trainer pays this at ~512px crops on cached tensors, the backbone pays cold image
  decode; (c) the resolution tail makes pre-tail epochs invalid soup ingredients (weights
  from different input resolutions may average, but their selection scores are
  incomparable and the pool would be polluted), so the pool logic needs tail-awareness.
  The lift does not justify the surface this round; revisit after A1 lands (a soup needs
  the trustworthy selection metric first anyway).

## Workstream B — shared-factory fixes

### B6. Stochastic depth in `create_model`

- `create_model(..., drop_path_rate: float | None = None)`: `None` -> do not pass the arg
  to timm (bit-identical to today); a float is forwarded (timm ConvNeXt's
  `drop_path_rate` — verified present in timm 1.0.27).
- `default_drop_path_rate(model_size) -> float` in `model.py`: atto/femto/pico/nano 0.1,
  tiny 0.2, base/large/huge 0.3.
- Backbone: config `drop_path_rate` — absent/None -> `default_drop_path_rate(model_size)`
  (ON for backbone builds: long-horizon trunk training is exactly the regime the ConvNeXt
  recipe evidences; the repo's regression evidence was a small ordinal GROUP fine-tune),
  explicit `0` disables. `BackboneTask.create_model` forwards it. `BackboneHeadsTask`
  keeps drop path OFF (frozen eval trunk; stochastic depth is inert in eval anyway — do
  not forward there).
- Group: `GroupTrainConfig.drop_path_rate: float = 0.0` (OFF by default — short fine-tunes,
  and the repo's own A/B caution). When > 0, `_create_or_warmstart_model` and the
  `GroupTask` resume-path `create_model` call forward it. DropPath is parameter-free so
  warm-start/state-dict compatibility is unaffected.

### B7. Weight-decay exclusions in `make_optimizer`

- `make_optimizer(model, *, llrd=False, llrd_decay=0.8, wd_exclusions=False)`:
  - Bare call stays EXACTLY one flat group (existing `test_flat_params_without_llrd` pins
    this) — the factory default is legacy; trainers opt in via config.
  - `wd_exclusions=True`, flat: two groups — `{params: ndim >= 2}` with the canonical
    0.01, `{params: ndim < 2, "weight_decay": 0.0, "name": "no_decay"}` (norm scales,
    biases, GRN gains are all 1-D in ConvNeXtV2).
  - `wd_exclusions=True` + `llrd=True`: each depth bucket splits into `<name>` and
    `<name>/no_decay` (wd 0.0) subgroups; the lr multiplier is shared within the depth.
    Implemented inside `make_optimizer` (or a `split_no_decay` kwarg on
    `build_llrd_param_groups` defaulting False — implementer's choice; the existing
    `test_llrd_param_groups_match_model_helper` must stay green, so the helper's default
    output must not change).
- Config: `GroupTrainConfig.wd_exclusions: bool = True`; backbone training_config
  `wd_exclusions: bool = True`. `gt._make_optimizer` passes it. Binary / multihead /
  dual-branch keep the legacy factory default this round (flip is a documented follow-up
  once their fingerprints carry the identity).
- Resume compatibility: the changed param-group layout must never load into old optimizer
  state. `training_state.make_fingerprint` gains an ADDITIVE
  `optimizer_identity: str | None = None` kwarg that folds an `"optimizer"` key into the
  fingerprint when set. `GroupTask.fingerprint_init` passes
  `optimizer_identity(llrd=cfg.llrd, llrd_decay=cfg.llrd_decay,
  wd_exclusions=cfg.wd_exclusions)`. Old group backups lack the key ->
  `fingerprint_matches` fails -> clean fresh start (resume_skipped), never a
  load_state_dict crash. Backbone: same via A4. Same-code pause->resume keeps matching
  (both sides carry the key), so `test_training_resume` equivalence holds.

### B8. Gradient clipping at the real step boundary

- `clip_gradients(parameters_or_module, max_norm: float) -> None` in
  `generic/optimizer.py` — thin wrapper over `torch.nn.utils.clip_grad_norm_` (accepts a
  module; clips `p for p in parameters() if p.grad is not None`).
- MONKEYPATCH SEAMS (the red tests patch these module globals — the loops must call the
  name through their own module namespace, house-style like `bb._evaluate`):
  - `bittrainer.group_trainer.clip_gradients` — called in `_train_one_epoch` INSIDE the
    accumulation-boundary block (`num_batches % accum == 0 or num_batches == total_steps`),
    immediately BEFORE `optimizer.step()`, only when `config.clip_grad_norm > 0`.
  - `bittrainer.generic.tasks.backbone_task.clip_gradients` — called in
    `BackboneTask.train_epoch` before `optimizer.step()` when enabled. (`BackboneHeadsTask`
    has its own `train_epoch`; wire it identically through the same
    `backbone_heads_task.clip_gradients` import — no red test pins it, but parity is
    expected and will be checked in review.)
- Config: `GroupTrainConfig.clip_grad_norm: float = 1.0`; backbone training_config
  `clip_grad_norm: float = 1.0`. `0` (or negative) disables. Default 1.0 per the audit —
  protective, standard for AdamW-family recipes, and deterministic (resume bit-exactness
  is unaffected because both sides of a resume clip identically).

## Workstream C — Engine split integrity

### C9. Near-duplicate-aware split assignment (pure logic this round)

`apps/engine/src/backend/backend/services/val_split.py` gains (additive, no DB):

- `NEARDUP_MAX_DISTANCE = 6` — matches the suite convention `are_similar(threshold=0.9)`
  == at most `floor(64 * 0.1) = 6` differing bits on the 64-bit dhash.
- `neardup_inherited_split(new_dhash: str | None, neighbours: Iterable[tuple[str | None,
  str | None]], *, max_distance: int = NEARDUP_MAX_DISTANCE) -> str | None`:
  1. Falsy `new_dhash` -> None.
  2. Keep neighbours whose split is `"train"`/`"val"`, whose dhash is truthy, and whose
     `hamming_distance(new_dhash, dhash) <= max_distance` (the shared
     `bitcrush_pyshared.db.hashing.hamming_distance`; mismatched-length hashes are
     inherently excluded because it returns the max distance).
  3. None qualify -> None (caller falls back to the ratio draw).
  4. All qualifying agree -> that split.
  5. Conflict (the cluster already straddles) -> the split of the strictly nearest
     neighbour; if the minimum distance itself carries both splits -> `"train"`
     (never add MORE near-dupes of train images into val — val inflation is the failure
     mode this fixes).
- `choose_split_neardup(train_count, val_count, *, min_val, ratio=DEFAULT_VAL_RATIO,
  new_dhash=None, neighbours=(), max_distance=NEARDUP_MAX_DISTANCE, rng=None) -> str`:
  inheritance takes ABSOLUTE precedence (before the first-image-to-train rule and the
  min-val top-up — a near-dup must land with its cluster, or the assignment itself creates
  the leak); otherwise delegates to `choose_split` unchanged.

DB wiring + backfill: SPEC ONLY, documented follow-up (below). No router/service changes
this round.

#### Follow-up spec: DB wiring (NOT implemented this round)

- Assignment site: `backend/services/group_labelling.py` (~line 1008) already holds the
  new image's `dhash` when it calls `choose_split`. Swap to `choose_split_neardup` with
  `neighbours` = `(dhash, split)` rows of the SAME GROUP's `GroupLabelledImage` where
  `dhash IS NOT NULL AND is_duplicate = 0` — group-wide, not class-wide: leakage is a
  property of the image, and near-dupes labelled into different classes still leak
  backgrounds/identity. Cheap query plan: exact `dhash ==` index probe first (exact dupes
  dominate), then a bounded candidate scan (the group's dhash column into memory, ~1e4
  rows max today; a BK-tree cache per group if profiling demands it).
- Same layer applies to `age_import.py` / `age_scanning.py` / `groups.py` call sites once
  the group path proves out.
- Backfill: one-off maintenance task that clusters existing `GroupLabelledImage` rows per
  group by dhash (union-find over pairs within 6 bits), finds straddling clusters, and
  moves the MINORITY side of each cluster to the majority split (tie -> train), rewriting
  `split`, moving files with sidecars (reuse `group_resplit.move_with_sidecars` plumbing)
  and flagging `val_resplit_pending_retrain` on touched groups (the moved images inflate
  the incumbent's val score exactly like the 0.1-ratio re-split did).

### C10. Frozen benchmark slice — SPEC ONLY

Goal: era-over-era comparability for backbone candidates and group promotions. Today every
score is computed on a moving val split, so "0.74 this month vs 0.71 last month" is not
evidence.

- **Capture**: a `benchmark_slices` table (Engine DB): `slice_id`, `created_at`,
  `frozen_by`, `scope` (`backbone` | `group:<name>`), and a member table of
  `(slice_id, content_hash, label_snapshot_json)`. A slice is built ONCE from the current
  val split (after C9 near-dup hygiene, so the slice is leak-free by construction),
  versioned, and NEVER mutated — label corrections create a NEW slice version
  (`slice_id` v2) rather than editing v1, because editing silently re-bases every historical
  number. Images are pinned by content_hash (survives file moves via the suite's
  content-hash index); a slice member whose file disappears is scored as missing and
  reported, not dropped.
- **Exclusion**: slice members are excluded from future TRAINING splits (a `frozen_val`
  split value or a join at audit time in `dataset_audit.py`) so the benchmark can never
  leak into a trunk.
- **Scoring**: after each backbone candidate export (and each group promotion gate), run
  the candidate on the newest slice version it is compatible with and store
  `(slice_id, run_id, metrics_json)` in a `benchmark_scores` table. Backbone metrics = the
  A1 panel (`binary_macro_f1`, `group_macro_f1`, `selection`, per-head F1/AP); group
  metrics = the existing shipped-decode panel. Era-over-era graphs join on `slice_id` so
  only like-for-like numbers are ever charted together.
- **Size**: cap per-head/per-class membership (e.g. 50 positives + 200 negatives per
  binary head; 50 per group class) so a full slice pass stays minutes on CPU-decode.

## Fingerprint / resume-compat summary

| Change | Fingerprint effect | Old backup outcome |
| --- | --- | --- |
| Backbone LLRD + WD exclusions | `fingerprint["optimizer"]` becomes e.g. `"Prodigy_adv+llrd0.8+nd"` | clean mismatch -> fresh start |
| Group WD exclusions | fingerprint gains `"optimizer"` key via `make_fingerprint(optimizer_identity=...)` | old backups lack the key -> clean mismatch |
| Grad clipping | none (deterministic, layout-neutral) | resume unaffected |
| Drop path | none (parameter-free) | resume unaffected |
| Augment/selection/calibration | none (no optimizer/model layout change) | resume unaffected; scores across eras not comparable (expected) |

## Wire-contract summary (Engine reads candidate metadata — additive only)

- `validation_metrics_json` keeps `binary/<c>` / `group/<g>` accuracy keys and GAINS
  `binary_f1/*`, `binary_ap/*`, `binary_support/*`, `group_f1/*`, `group_val_support/*`,
  `binary_macro_f1`, `group_macro_f1`, `selection`.
- `validation_score` semantics change from mean-accuracy to the `selection` aggregate
  (still a single float in [0,1]; Engine only displays/compares it within a run family —
  called out for the coordinator, not a schema change).
- NEW metadata keys: `binary_thresholds_json`, `binary_temperatures_json` (optional).
- NEW training_config keys (all with defaults, absent == default): `calibrate_thresholds`,
  `calibrate_temperature`, `use_random_resized_crop`, `rrc_scale_min`, `randaugment_n`,
  `randaugment_m`, `random_erasing_p`, `group_label_smoothing`, `loss_reduction`,
  `positive_cap_mode`, `llrd`, `llrd_decay`, `wd_exclusions`, `clip_grad_norm`,
  `drop_path_rate`, `batch_size: "auto"|0`.

## Risks

- `_batch_loss` reduction default flip (task-level "mean") changes training dynamics for
  every new backbone run — intended, Prodigy adapts, but the first GPU run should be
  A/B'd (see final report recommendations).
- Per-group `weight_decay: 0.0` relies on Prodigy_adv honouring per-group values
  (standard `Optimizer.add_param_group` semantics; `cautious_wd` reads the group's wd).
  The WD test asserts the group values; if Prodigy ignores them the test still passes —
  reviewer must eyeball `adv_optm`'s step loop once.
- RandAugment on PIL inputs needs `transforms.RandAugment` before `ToTensor` (uint8);
  ordering inside `_train_transform` is implementation freedom but the val transform is
  pinned by test.
- Group-side defaults changed by this round: `wd_exclusions=True`, `clip_grad_norm=1.0`
  (drop path stays OFF). Both are deliberate and config-escapable.

## Ordered implementation sequence

1. B7 `make_optimizer` exclusions + `optimizer_identity` + `make_fingerprint` kwarg
   (test_optimizer_wd_exclusions).
2. A4 LLRD bucketing fix + backbone fingerprint identity + task llrd config
   (test_llrd_wrapper_prefixes).
3. B6 drop path factory + defaults table + plumbing (test_drop_path_factory).
4. B8 clip_gradients helper + both loop wirings (test_grad_clipping).
5. A3 transforms + `_batch_loss` smoothing/reduction + plan-mode knob
   (test_backbone_augment_parity).
6. A1 collection + metrics + selection (test_backbone_selection_metrics).
7. A2 threshold/temperature tuning + metadata export (test_backbone_calibration_export).
8. A5 autobatch opt-in (covered inside test_llrd_wrapper_prefixes' run? no — covered by
   test_backbone_calibration_export's config? NO: autobatch has its own test inside
   test_grad_clipping? — see test inventory: `test_backbone_augment_parity.py` carries the
   autobatch wiring test).
9. C9 Engine `val_split.py` additive functions (test_val_split_neardup).
10. Run the full named test set (new + related existing) and paste output.

## Test inventory (red tests, authored by the design lead)

| File | Locks in |
| --- | --- |
| `test_backbone_selection_metrics.py` | `_backbone_metrics` per-head F1/AP, zero-support exclusion, aggregates, legacy keys, `selection_score` fallback |
| `test_backbone_calibration_export.py` | `_tune_binary_thresholds` F1-optimality + fallback, `_fit_binary_temperature` NLL descent, metadata export end-to-end |
| `test_backbone_augment_parity.py` | train-transform defaults + gating, val Resize+CenterCrop determinism, `_batch_loss` smoothing/reduction, `positive_cap_mode`, autobatch opt-in wiring |
| `test_llrd_wrapper_prefixes.py` | wrapper-prefix bucketing, plain-model invariance, `optimizer_identity`, backbone fingerprint key, task passes `llrd=True` |
| `test_optimizer_wd_exclusions.py` | no-decay split flat + LLRD, legacy bare-call layout, group config field + pass-through, `make_fingerprint(optimizer_identity=...)` |
| `test_drop_path_factory.py` | `default_drop_path_rate` table, `create_model` arg, default = bit-identical legacy, backbone task default ON / 0 off, group config default OFF |
| `test_grad_clipping.py` | `clip_gradients` helper, group boundary-only clipping under grad accumulation, backbone wiring + disable at 0 |
| Engine `test_val_split_neardup.py` | `neardup_inherited_split` rules 1-5, `choose_split_neardup` precedence + fallback |

## Baseline (must stay green — the only existing tests the implementer runs)

BitTrainer: `test_backbone_generic.py`, `test_backbone_trainer.py`,
`test_backbone_sampling.py`, `test_backbone_resolution.py`, `test_backbone_soft_negatives.py`,
`test_backbone_heads_task.py`, `test_optimizer_factory.py`, `test_group_task_parity.py`,
`test_training_resume.py`, `test_group_loss_extraction.py`.
Engine: `apps/engine/tests/backend/test_val_split.py`.

Command shape (BitTrainer — the clone's venv has no pytest; use the suite venv with
PYTHONPATH shadowing so the CLONE is what runs, per the house harness):
`cd "/f/Projects/Other Tools/BItTrainer" && CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="F:/Projects/Other Tools/BItTrainer" "/f/Projects/Bitcrush/Bitcrush Suite/.venv/Scripts/python.exe" -m pytest bittrainer/tests/<file> -x -q`
Engine (suite root; the backend package needs PYTHONPATH):
`cd "/f/Projects/Bitcrush/Bitcrush Suite" && CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="F:/Projects/Bitcrush/Bitcrush Suite/apps/engine/src/backend" .venv/Scripts/python.exe -m pytest "apps/engine/tests/backend/test_val_split_neardup.py" -x -q`
