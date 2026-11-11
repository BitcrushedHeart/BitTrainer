"""Check this round's work on the rented GPU, inside a time budget (Bitcrush ISSUE-1655).

Nothing built for the Azure T4 has run on a GPU. This runs all of it there, in
stages, on generated images, and writes down what passed and how fast it went::

    python -m bittrainer.tools.vm_check --work /mnt/vmcheck
    python -m bittrainer.tools.vm_check --work /mnt/vmcheck --list      # the plan
    python -m bittrainer.tools.vm_check --work /mnt/vmcheck --pack /data/pack

The checks that decide whether the real run can start come first: the GPU-only
tests, a model learning in this card's precision, the real weight conversion,
training / pausing / resuming from a pack, the switch to the 384px tail at its
own batch size (and a tail that cannot fit being refused before anything
trains), and distillation with a teacher wider than its student. Then the
group trainers (Bitcrush ISSUE-1690), through ``bittrainer.tools.group_proof``:
head-only, a scoped export with private names and a sourceless full update in
this card's precision for Base (compiled), nano and tiny at 512px, an oversize
batch refused before it trains, and a paused update resumed. Like a real run,
each starts from a trained trunk the series made: the Base teacher, the
distilled nano, or an Apache V1 conversion; never random weights (which
collapsed the fp16 scaler on the T4) and never timm's ConvNeXt V2 weights
(non-commercial). A group run whose scaler collapsed fails. Speed comparisons
come last and are the first thing dropped when ``--budget-minutes`` (default
120) runs short. Estimates are what the T4 took in rehearsal R1 (2026-11-10).

It is safe to stop and run again: a stage that passed is not repeated
(``--redo`` repeats everything). Everything it writes goes under ``--work``:

    report.json   every stage's result and numbers
    summary.txt   the same, readable
    logs/         each stage's full output
    packs/ runs/  the generated packs and the training runs

No real image is needed. ``--pack`` adds one stage that trains briefly on part of
a real pack, for the speed that matters: real images and the real head count.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

_TESTS = Path(__file__).resolve().parents[1] / "tests"
_ROUND_TESTS = (
    "test_backbone_amp.py",
    "test_backbone_head_vectorised.py",
    "test_backbone_proof.py",
    "test_backbone_pack_train.py",
    "test_backbone_distill.py",
    "test_convnext_v1_init.py",
    "test_lean_grn.py",
    "test_vm_check.py",
    "test_backbone_tail_batch.py",
    "test_backbone_skipped_steps.py",
    "test_backbone_auto_batch.py",
    "test_backbone_run_record.py",
    "test_backbone_label_policy.py",
    "test_backbone_geometry.py",
    "test_backbone_validation_report.py",
    "test_backbone_status.py",
    # The group trainers on fp16 cards (Bitcrush ISSUE-1690).
    "test_group_amp.py",
    "test_group_lean_grn.py",
    "test_group_preflight.py",
    "test_group_proof.py",
    "test_autobatch.py",
    "test_cloud_export_private_names.py",
)
# The files with tests that only run on a GPU (named ..._real_gpu, or the ones
# that check the precision matches the card).
_GPU_TESTS = (
    "test_backbone_amp.py",
    "test_backbone_tail_batch.py",
    "test_backbone_skipped_steps.py",
    "test_backbone_auto_batch.py",
    "test_group_amp.py",
    "test_group_lean_grn.py",
    "test_group_preflight.py",
)
_OTHER_TESTS = (
    "test_backbone_generic.py",
    "test_backbone_trainer.py",
    "test_backbone_heads_task.py",
    "test_backbone_resolution.py",
    "test_backbone_augment_parity.py",
    "test_backbone_calibration_export.py",
    "test_backbone_sampling.py",
    "test_backbone_selection_metrics.py",
    "test_backbone_soft_negatives.py",
    "test_backbone_init.py",
    "test_backbone_hash.py",
    "test_generic_trainer.py",
    # Outside the package's own test folder. It builds the optimiser without a
    # run context, which is how the preflight's first version was caught.
    "../../tests/test_backbone_trainer_parity.py",
)


@dataclass
class Result:
    status: str  # passed | failed | skipped
    numbers: dict = field(default_factory=dict)
    note: str = ""
    seconds: float = 0.0


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    minutes: float  # estimate on a T4
    run: Callable[["Context"], Result]
    needs: tuple[str, ...] = ()
    optional: bool = False  # a speed comparison: dropped first when time is short


class Context:
    """Where a run of the plan keeps its files, and how it tells the time."""

    def __init__(
        self,
        work: Path,
        *,
        pack: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        python: str = sys.executable,
    ) -> None:
        self.work = Path(work)
        self.pack = Path(pack) if pack else None
        self.clock = clock
        self.python = python
        self.report: dict = {"stages": {}}


def _log(message: str) -> None:
    print(f"[check] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Running the plan                                                            #
# --------------------------------------------------------------------------- #


def run_plan(
    stages: list[Stage],
    context: Context,
    *,
    budget_minutes: float,
    only: set[str] | None = None,
    skip: set[str] | None = None,
    redo: bool = False,
) -> dict:
    """Run *stages* in order and return the report (also saved after each one).

    A stage is skipped when something it needs did not pass, when it is a speed
    comparison that no longer fits the budget, or when the budget is spent. A
    stage that passed in an earlier run of the same ``--work`` is kept, not
    repeated. A stage that raises is a failure of that stage, not of the plan.
    """
    context.work.mkdir(parents=True, exist_ok=True)
    report_path = context.work / "report.json"
    if report_path.is_file() and not redo:
        context.report = json.loads(report_path.read_text(encoding="utf-8"))
    results: dict = context.report.setdefault("stages", {})
    started = context.clock()

    def _save(name: str, result: Result) -> None:
        results[name] = asdict(result)
        report_path.write_text(json.dumps(context.report, indent=1), encoding="utf-8")

    for stage in stages:
        if (only and stage.name not in only) or (skip and stage.name in skip):
            continue
        if results.get(stage.name, {}).get("status") == "passed":
            continue
        used = (context.clock() - started) / 60
        unmet = [n for n in stage.needs if results.get(n, {}).get("status") != "passed"]
        if unmet:
            _save(stage.name, Result("skipped", note=f"needs {', '.join(unmet)}"))
        elif used >= budget_minutes:
            _save(stage.name, Result("skipped", note="the time budget is spent"))
        elif stage.optional and used + stage.minutes > budget_minutes:
            _save(stage.name, Result("skipped", note="no longer fits the time budget"))
        else:
            _log(f"{stage.name}: {stage.title} (about {stage.minutes:g} min)")
            began = context.clock()
            try:
                result = stage.run(context)
            except Exception as exc:  # a stage's own failure, reported as one
                result = Result("failed", note=f"{type(exc).__name__}: {exc}")
            result.seconds = round(context.clock() - began, 1)
            _save(stage.name, result)
            _log(f"{stage.name}: {result.status} in {result.seconds:g}s {result.note}")
    return context.report


def summary_lines(stages: list[Stage], report: dict) -> list[str]:
    """One readable line per stage: status, time, title, then its numbers."""
    labels = {"passed": "passed ", "failed": "FAILED ", "skipped": "skipped"}
    lines = []
    for stage in stages:
        result = report.get("stages", {}).get(stage.name)
        if result is None:
            lines.append(f"not run  {stage.name}: {stage.title}")
            continue
        line = f"{labels.get(result['status'], result['status'])}  {stage.name}: {stage.title}"
        if result.get("seconds"):
            line += f" [{result['seconds']:g}s]"
        if result.get("note"):
            line += f" ({result['note']})"
        lines.append(line)
        numbers = ", ".join(
            f"{key}={value}" for key, value in result["numbers"].items()
        )
        if numbers:
            lines.append(f"         {numbers}")
    return lines


# --------------------------------------------------------------------------- #
# Reading what a stage produced                                               #
# --------------------------------------------------------------------------- #


def last_json(output: str) -> dict | None:
    """The last pretty-printed JSON object in *output* (the proof tool's report)."""
    lines = output.splitlines()
    for end in range(len(lines) - 1, -1, -1):
        if lines[end] != "}":
            continue
        for start in range(end, -1, -1):
            if lines[start] == "{":
                try:
                    return json.loads("\n".join(lines[start : end + 1]))
                except ValueError:
                    break
    return None


def run_numbers(out: Path) -> dict:
    """What a pack training run reported: speed, precision, epochs, resume,
    the sizes it trained at, what its preflight measured and what it skipped."""
    numbers: dict = {}
    progress = Path(out) / "progress.jsonl"
    if not progress.is_file():
        return numbers
    speeds = []
    epochs = 0
    phases: list[str] = []
    for raw in progress.read_text(encoding="utf-8").splitlines():
        try:
            frame = json.loads(raw)
        except ValueError:
            continue
        if frame.get("stage") == "preparing" and "amp_dtype" in frame:
            numbers["precision"] = frame.get("amp_dtype")
            numbers["grad_scaler"] = frame.get("grad_scaler")
        elif frame.get("stage") == "training" and "loss" in frame:
            epochs += 1
            if frame.get("images_per_second"):
                speeds.append(frame["images_per_second"])
            if frame.get("loss_terms"):
                numbers["loss_terms"] = frame["loss_terms"]
            if frame.get("batch_size"):
                phase = f"{frame.get('image_size')}px x{frame['batch_size']}"
                if phase not in phases:
                    phases.append(phase)
        elif frame.get("stage") == "preflight":
            checked = numbers.setdefault("preflight_mb", {})
            checked[frame.get("phase")] = frame.get("peak_vram_mb")
        elif frame.get("stage") == "warning":
            numbers["warnings"] = numbers.get("warnings", 0) + 1
        elif frame.get("type") == "autobatch":
            tail = frame.get("phase") == "tail"
            key = "auto_tail_batch_size" if tail else "auto_batch_size"
            numbers[key] = frame.get("batch_size")
        elif frame.get("type") == "training_resumed":
            numbers["resumed"] = True
    numbers["epochs"] = epochs
    if phases:
        numbers["phases"] = phases
    if speeds:
        numbers["images_per_second"] = round(max(speeds), 1)
    result = Path(out) / "result.json"
    if result.is_file():
        saved = json.loads(result.read_text(encoding="utf-8"))
        for key in ("validation_score", "peak_vram_mb", "skipped_steps", "grad_scale"):
            if saved.get(key) is not None:
                numbers[key] = saved[key]
    return numbers


# --------------------------------------------------------------------------- #
# A pack of generated images                                                  #
# --------------------------------------------------------------------------- #


def write_synthetic_pack(
    root: Path,
    *,
    images: int,
    image_size: int,
    extra_heads: int = 0,
    labels_per_image: int = 30,
    seed: int = 0,
    geometry_sensitive: bool = False,
) -> Path:
    """A format-1 pack drawn by the proof tool: two concepts and one group.

    ``geometry_sensitive`` lists the group in the manifest, so every image with
    a group label trains with colour-only augmentation.

    ``extra_heads`` adds that many more yes/no heads, each labelled on a sparse
    random share of the images (about ``labels_per_image`` answers per image in
    all) and each repeating one of the two real signals, so they can be learned.
    That is the shape of a real pool: hundreds of heads, a few dozen answers per
    image.
    """
    from bittrainer.tools.backbone_proof import make_fixture_dataset

    root = Path(root)
    drawn = root / "_drawn"
    rng = random.Random(seed)
    rate = min(1.0, labels_per_image / extra_heads) if extra_heads else 0.0
    classes = {"side_by_side": "k1", "stacked": "k2", "centre_blob": "k3"}
    lines = []
    records = make_fixture_dataset(
        drawn, count=images, image_size=image_size, seed=seed
    )
    for record in records:
        digest = record["content_hash"]
        target = root / "images" / digest[:2] / f"{digest}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(record["file_paths"][0], target)
        answers = {name: int(a == "positive") for name, a in record["binary"].items()}
        binary = {
            head: answers[name]
            for head, name in (("c1", "striped"), ("c2", "ringed"))
            if name in answers
        }
        for index in range(extra_heads):
            signal = "striped" if index % 2 == 0 else "ringed"
            if signal in answers and rng.random() < rate:
                binary[f"c{101 + index}"] = answers[signal]
        line: dict = {"h": digest}
        if binary:
            line["b"] = binary
        if record["groups"]:
            line["g"] = {"g1": classes[record["groups"]["layout"]]}
        if len(line) > 1:
            lines.append(json.dumps(line, separators=(",", ":")))
    shutil.rmtree(drawn, ignore_errors=True)
    lines.sort()
    (root / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The group here is a layout (side by side, stacked, centre blob): position
    # is its answer, so it is the kind of head the export marks.
    manifest = {
        "format": 1,
        "images": len(lines),
        "short_side": image_size,
        "geometry_sensitive": ["g1"] if geometry_sensitive else [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# Running commands                                                            #
# --------------------------------------------------------------------------- #


def _command(
    context: Context, name: str, argv: list[str], minutes: float
) -> tuple[int, str]:
    """Run *argv* to completion, its output into ``logs/<name>.log``."""
    log = context.work / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    limit = max(300.0, minutes * 60 * 3)
    with log.open("w", encoding="utf-8") as handle:
        try:
            done = subprocess.run(  # noqa: S603 - our own interpreter and modules
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=limit,
                check=False,
            )
            code = done.returncode
        except subprocess.TimeoutExpired:
            code = -1
    return code, log.read_text(encoding="utf-8", errors="replace")


def _module(context: Context, module: str, *args: str) -> list[str]:
    return [context.python, "-m", module, *args]


def _fresh_run_folder(context: Context, name: str) -> Path:
    out = context.work / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    return out


def _train(
    context: Context, name: str, pack: Path, args: list[str], minutes: float
) -> tuple[Result, Path]:
    """One ``backbone_pack_train`` run into a fresh folder under ``runs/``."""
    out = _fresh_run_folder(context, name)
    argv = _module(
        context,
        "bittrainer.tools.backbone_pack_train",
        "--pack", str(pack), "--out", str(out), *args,
    )  # fmt: skip
    code, _output = _command(context, name, argv, minutes)
    numbers = run_numbers(out)
    if code != 0:
        return Result("failed", numbers, f"exit {code}; see logs/{name}.log"), out
    return Result("passed", numbers), out


def pause_then_resume(
    context: Context,
    name: str,
    argv: list[str],
    out: Path,
    *,
    poll_seconds: float = 1.0,
    limit_seconds: float = 1500.0,
) -> Result:
    """Start a run, pause it through its ``PAUSE`` file once it is training,
    then run the same command again and expect it to resume and finish."""
    out = Path(out)
    log = context.work / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT)  # noqa: S603
        deadline = time.monotonic() + limit_seconds
        asked = False
        while process.poll() is None and time.monotonic() < deadline:
            if (
                not asked
                and progress.is_file()
                and "training_step"
                in progress.read_text(encoding="utf-8", errors="replace")
            ):
                (out / "PAUSE").touch()
                asked = True
            time.sleep(poll_seconds)
        if process.poll() is None:
            process.kill()
            process.wait()
            return Result("failed", note="the run did not stop within its time limit")
        if process.returncode != 3:
            return Result(
                "failed",
                run_numbers(out),
                f"expected the paused run to exit 3, got exit {process.returncode}",
            )
        handle.write("\n--- resuming ---\n")
        handle.flush()
        try:
            again = subprocess.run(  # noqa: S603
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=limit_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Result("failed", note="the resumed run did not finish in time")
    numbers = run_numbers(out)
    if again.returncode != 0:
        return Result(
            "failed", numbers, f"the resumed run ended with exit {again.returncode}"
        )
    if not numbers.get("resumed"):
        return Result(
            "failed", numbers, "the second run started over instead of resuming"
        )
    return Result("passed", numbers)


# --------------------------------------------------------------------------- #
# The stages                                                                  #
# --------------------------------------------------------------------------- #

_BASE = ["--size", "base", "--init", "apache", "--batch-size", "64", "--workers", "4"]
_NANO = ["--size", "nano", "--init", "apache", "--batch-size", "64", "--workers", "4"]


def _small_pack(context: Context) -> Path:
    return context.work / "packs" / "small"


def _teacher_file(context: Context) -> str:
    return str(context.work / "runs" / "teacher" / "backbone_candidate.safetensors")


def _distilled_nano(context: Context) -> str:
    return str(context.work / "runs" / "distill" / "backbone_candidate.safetensors")


def _converted(size: str) -> Callable[[Context], str]:
    """The Apache V1 trunk the ``convert_*`` stages write for *size*."""

    def _path(context: Context) -> str:
        return str(context.work / "trunks" / f"{size}.safetensors")

    return _path


def _env(context: Context) -> Result:
    import torch

    from bittrainer.amp import native_bf16

    if not torch.cuda.is_available():
        return Result("failed", note="torch sees no CUDA device")
    card = torch.cuda.get_device_properties(0)
    return Result(
        "passed",
        {
            "gpu": card.name,
            "capability": f"{card.major}.{card.minor}",
            "vram_gb": round(card.total_memory / 2**30, 1),
            "torch": torch.__version__,
            "precision": "bfloat16" if native_bf16(torch.device("cuda")) else "float16",
        },
    )


def _pytest(context: Context, name: str, files, minutes: float, *extra: str):
    argv = _module(
        context,
        "pytest",
        *[str(_TESTS / f) for f in files],
        "-q",
        "-p",
        "no:cacheprovider",
        *extra,
    )
    code, output = _command(context, name, argv, minutes)
    tail = next(
        (line for line in reversed(output.splitlines()) if " passed" in line), ""
    )
    return code, output, {"result": tail.strip(" =")}


def _gpu_tests(context: Context) -> Result:
    code, _output, numbers = _pytest(
        context, "gpu_tests", _GPU_TESTS, 2,
        "-k", "real_gpu or matches_the_card",
    )  # fmt: skip
    # Picked by name, so a skipped one means the GPU was not seen.
    ran = numbers["result"]
    if code != 0 or " passed" not in ran or " skipped" in ran:
        return Result("failed", numbers, "every GPU test must run and pass (not skip)")
    return Result("passed", numbers)


def _round_tests(context: Context) -> Result:
    code, _output, numbers = _pytest(context, "round_tests", _ROUND_TESTS, 9)
    return Result("passed" if code == 0 else "failed", numbers)


def _other_tests(context: Context) -> Result:
    code, _output, numbers = _pytest(context, "other_tests", _OTHER_TESTS, 7)
    return Result("passed" if code == 0 else "failed", numbers)


def _proof(context: Context, name: str, minutes: float, *args: str) -> Result:
    argv = _module(
        context, "bittrainer.tools.backbone_proof", "--device", "cuda", *args
    )
    code, output = _command(context, name, argv, minutes)
    report = last_json(output) or {}
    numbers = {
        "images_per_second": round(report.get("best_images_per_second") or 0.0, 1),
        "precision": report.get("amp_dtype"),
        "grad_scaler": report.get("grad_scaler"),
        "peak_vram_mb": report.get("peak_vram_mb"),
        "skipped_steps": report.get("skipped_steps"),
    }
    if report.get("lean_grn") is False:
        numbers["grn"] = "stock"
    if "weakest_family_f1" in report and report.get("min_score") is not None:
        numbers["weakest_family_f1"] = round(report["weakest_family_f1"] or 0.0, 3)
    if code != 0 or not report.get("passed"):
        return Result("failed", numbers, f"exit {code}; see logs/{name}.log")
    return Result("passed", numbers)


def _learn_fp16(context: Context) -> Result:
    # The proof's own defaults, with its learning floor on: both head families
    # must learn in whatever precision this card trains in.
    return _proof(context, "learn_fp16", 2)


def _speed(name: str, *args: str) -> Callable[[Context], Result]:
    def _run(context: Context) -> Result:
        return _proof(
            context, name, 5,
            "--images", "4000", "--epochs", "3", "--workers", "4", "--min-score", "-1", *args,
        )  # fmt: skip

    return _run


def _convert(*sizes: str) -> Callable[[Context], Result]:
    def _run(context: Context) -> Result:
        numbers = {}
        for size in sizes:
            target = context.work / "trunks" / f"{size}.safetensors"
            argv = _module(
                context, "bittrainer.tools.convnext_v1_trunk",
                "--size", size, "--out", str(target),
            )  # fmt: skip
            code, output = _command(context, f"convert_{size}", argv, 2)
            found = re.search(r"features match to ([0-9.e+-]+)", output)
            numbers[f"{size}_difference"] = (
                found.group(1).rstrip(".") if found else None
            )
            if code != 0:
                return Result("failed", numbers, f"{size}: exit {code}")
        return Result("passed", numbers)

    return _run


def _synth_packs(context: Context) -> Result:
    small = _small_pack(context)
    many = context.work / "packs" / "many_heads"
    for pack, extra in ((small, 0), (many, 900)):
        if not (pack / "manifest.json").is_file():
            shutil.rmtree(pack, ignore_errors=True)
            write_synthetic_pack(pack, images=4000, image_size=256, extra_heads=extra)
    geometry = context.work / "packs" / "geometry"
    if not (geometry / "manifest.json").is_file():
        shutil.rmtree(geometry, ignore_errors=True)
        write_synthetic_pack(
            geometry, images=1500, image_size=256, geometry_sensitive=True
        )
    return Result("passed", {"images": 4000, "many_heads": 902, "geometry": 1500})


def _candidate_flag(out: Path, key: str) -> str | None:
    from safetensors import safe_open

    with safe_open(
        str(out / "backbone_candidate.safetensors"), framework="pt"
    ) as handle:
        return (handle.metadata() or {}).get(key)


def _geometry(context: Context) -> Result:
    # Every image here carries a label for a listed head, so every image takes
    # the colour-only path; the candidate must say geometry was protected.
    result, out = _train(
        context, "geometry", context.work / "packs" / "geometry",
        [*_NANO, "--epochs", "2", "--max-steps", "60"], 2,
    )  # fmt: skip
    if result.status == "passed":
        flag = _candidate_flag(out, "geometry_safe_augmentation")
        result.numbers["geometry_safe_augmentation"] = flag
        if flag != "true":
            return Result(
                "failed",
                result.numbers,
                "the candidate does not say geometry was protected",
            )
    return result


def _unlabelled_option(name: str, *flags: str) -> Callable[[Context], Result]:
    """One of the research's options for unlabelled images, on the small pack."""

    def _run(context: Context) -> Result:
        result, _out = _train(
            context, name, _small_pack(context),
            [*_NANO, *flags, "--epochs", "3", "--max-steps", "150"], 2.5,
        )  # fmt: skip
        return result

    return _run


def _teacher(context: Context) -> Result:
    result, _out = _train(
        context, "teacher", _small_pack(context),
        [*_BASE, "--epochs", "2", "--progress-every", "20"], 5,
    )  # fmt: skip
    return result


def _pause_resume(context: Context) -> Result:
    out = _fresh_run_folder(context, "pause_resume")
    argv = _module(
        context, "bittrainer.tools.backbone_pack_train",
        "--pack", str(_small_pack(context)), "--out", str(out),
        *_BASE, "--epochs", "3", "--progress-every", "5",
    )  # fmt: skip
    return pause_then_resume(context, "pause_resume", argv, out)


# A short run shaped like the real one: an epoch at 224px, then a 384px tail.
_TAIL = [
    "--epochs", "2", "--limit", "400", "--progress-every", "5",
    "--finetune-image-size", "384", "--finetune-epochs", "1",
]  # fmt: skip


def tail_problem(numbers: dict) -> str:
    """What is wrong with a run that should have trained Base at 224px and
    batch 64, then at 384px and batch 16; empty when nothing is."""
    if numbers.get("phases") != ["224px x64", "384px x16"]:
        return (
            f"expected 224px at batch 64, then 384px at batch 16; the run "
            f"trained {numbers.get('phases')}"
        )
    if set(numbers.get("preflight_mb") or {}) != {"main", "tail"}:
        return "the preflight did not check both phases before training"
    return ""


def _tail(context: Context) -> Result:
    result, _out = _train(
        context, "tail", _small_pack(context),
        [*_BASE, *_TAIL, "--finetune-batch-size", "16"], 3,
    )  # fmt: skip
    problem = tail_problem(result.numbers) if result.status == "passed" else ""
    return Result("failed", result.numbers, problem) if problem else result


def refusal_problem(code: int, output: str, progress: str) -> str:
    """What is wrong with a run the preflight should have refused; empty when
    it was refused as it should be: exit 2, the tail named, nothing trained."""
    if code != 2:
        return f"expected exit 2 (refused), got exit {code}"
    if "REFUSED" not in output or "tail" not in output:
        return "the refusal does not name the tail"
    if "training_step" in progress:
        return "the run trained before it was refused"
    return ""


def _preflight_refuses(context: Context) -> Result:
    # 384px at batch 64 needs about twice this card. The run has to say so
    # before it trains anything, not at the switch.
    out = _fresh_run_folder(context, "preflight_refuses")
    argv = _module(
        context, "bittrainer.tools.backbone_pack_train",
        "--pack", str(_small_pack(context)), "--out", str(out),
        *_BASE, *_TAIL, "--finetune-batch-size", "64",
    )  # fmt: skip
    code, output = _command(context, "preflight_refuses", argv, 1.5)
    progress = out / "progress.jsonl"
    problem = refusal_problem(
        code, output, progress.read_text("utf-8") if progress.is_file() else ""
    )
    numbers = run_numbers(out)
    return Result("failed", numbers, problem) if problem else Result("passed", numbers)


def _distill(context: Context) -> Result:
    result, out = _train(
        context, "distill", _small_pack(context),
        [*_NANO, "--teacher", _teacher_file(context), "--epochs", "2"], 4,
    )  # fmt: skip
    if result.status != "passed":
        return result
    from safetensors import safe_open

    with safe_open(
        str(out / "backbone_candidate.safetensors"), framework="pt"
    ) as handle:
        metadata = handle.metadata() or {}
    if "distilled_from" not in metadata:
        return Result(
            "failed", result.numbers, "the candidate does not say it was distilled"
        )
    if not result.numbers.get("loss_terms"):
        return Result("failed", result.numbers, "no distillation terms were reported")
    return result


def _distill_pause_resume(context: Context) -> Result:
    out = _fresh_run_folder(context, "distill_pause_resume")
    argv = _module(
        context, "bittrainer.tools.backbone_pack_train",
        "--pack", str(_small_pack(context)), "--out", str(out),
        *_NANO, "--teacher", _teacher_file(context), "--epochs", "3", "--progress-every", "5",
    )  # fmt: skip
    return pause_then_resume(context, "distill_pause_resume", argv, out)


def _distill_channels_last(context: Context) -> Result:
    result, _out = _train(
        context, "distill_channels_last", _small_pack(context),
        [*_NANO, "--teacher", _teacher_file(context), "--channels-last",
         "--epochs", "2", "--max-steps", "60"], 2,
    )  # fmt: skip
    return result


def _soft_negatives(context: Context) -> Result:
    result, _out = _train(
        context, "soft_negatives", _small_pack(context),
        [*_NANO, "--soft-negatives", "--epochs", "2", "--max-steps", "100"], 2,
    )  # fmt: skip
    return result


def auto_batch_problem(numbers: dict) -> str:
    """What is wrong with the sizes ``--batch-size auto`` chose for Base on a
    16 GB card; empty when nothing is. Batch 64 is known to fit at 224px, and
    the first version chose 8."""
    main, tail = numbers.get("auto_batch_base"), numbers.get("auto_batch_base_tail")
    if not main or main < 48:
        return f"Base at 224px was sized at {main}; 64 is known to fit"
    if not tail or tail < 8 or tail >= main:
        return f"the 384px tail was sized at {tail}, beside {main} at 224px"
    return ""


def _auto_batch(context: Context) -> Result:
    # Base with a tail, each phase sized on its own and then trained at that
    # size. Then a nano with a Base teacher loaded: its forward pass is part of
    # every step that is measured.
    numbers = {}
    runs = (
        ("auto_batch_base", ["--size", "base", "--init", "apache", *_TAIL]),
        (
            "auto_batch_distill",
            [*_NANO[:4], "--teacher", _teacher_file(context),
             "--epochs", "1", "--max-steps", "30"],
        ),
    )  # fmt: skip
    for name, args in runs:
        result, _out = _train(
            context, name, _small_pack(context),
            [*args, "--workers", "4", "--batch-size", "auto"], 2.5,
        )  # fmt: skip
        numbers[name] = result.numbers.get("auto_batch_size")
        numbers[f"{name}_tail"] = result.numbers.get("auto_tail_batch_size")
        numbers[f"{name}_vram_mb"] = result.numbers.get("peak_vram_mb")
        if result.status != "passed":
            return Result("failed", numbers, f"{name}: {result.note}")
    problem = auto_batch_problem(numbers)
    return Result("failed" if problem else "passed", numbers, problem)


def _many_heads(context: Context) -> Result:
    result, _out = _train(
        context, "many_heads", context.work / "packs" / "many_heads",
        [*_BASE, "--epochs", "3", "--max-steps", "150"], 5,
    )  # fmt: skip
    plain = context.report["stages"].get("teacher", {}).get("numbers", {})
    speed, baseline = (
        result.numbers.get("images_per_second"),
        plain.get("images_per_second"),
    )
    if speed and baseline:
        result.numbers["speed_against_three_heads"] = round(speed / baseline, 2)
    return result


def _real_pack(context: Context) -> Result:
    if context.pack is None:
        return Result("skipped", note="no --pack was given")
    result, _out = _train(
        context,
        "real_pack",
        context.pack,
        [*_BASE, "--limit", "5000", "--epochs", "1"],
        6,
    )
    return result


# --------------------------------------------------------------------------- #
# The group trainers (Bitcrush ISSUE-1690)                                    #
# --------------------------------------------------------------------------- #

_GROUP_512 = ("--bucket-size", "512", "--workers", "4")
_GROUP_SMALL = (*_GROUP_512, "--images-per-class", "24", "--no-compile")


def group_numbers(report: dict) -> dict:
    """What a group proof run reported, for the summary."""
    numbers = {
        "images_per_second": round(report.get("best_images_per_second") or 0.0, 1),
        "precision": report.get("amp_dtype"),
        "grad_scaler": report.get("grad_scaler"),
        "lean_grn": report.get("lean_grn"),
        "compiled": report.get("compiled"),
        "batch_size": report.get("batch_size_used") or report.get("batch_size"),
        "peak_vram_mb": report.get("peak_vram_mb"),
        "preflight_mb": report.get("preflight_mb"),
        "skipped_steps": report.get("skipped_steps"),
        "grad_scale": report.get("grad_scale"),
    }
    for key in ("refused", "resumed"):
        if report.get(key):
            numbers[key] = report[key]
    return numbers


def group_problem(report: dict, args: tuple[str, ...], *, native_bf16: bool) -> str:
    """What is wrong with a group proof run, on a card that does (or does not)
    have native bf16; empty when nothing is. The proof cannot know the card: a
    T4 must have trained in fp16 through a GradScaler, with the lean GRN layer
    unless the stock one was asked for, and a compile that was asked for must
    not have fallen back to eager. And the scaler must have held, by the
    trainer's own rule (``group_task.scaler_collapse``): a scale below 1, or
    more than half of an epoch's steps (at least 3) skipped after the scaler
    settled, is a run that barely trained."""
    if "--expect-refusal" in args:
        return ""
    want = "bfloat16" if native_bf16 else "float16"
    if report.get("amp_dtype") != want:
        return f"trained in {report.get('amp_dtype')}; this card should train in {want}"
    if report.get("grad_scaler") is native_bf16:
        return f"GradScaler {'on' if native_bf16 else 'off'} in {want}"
    lean = not native_bf16 and "--stock-grn" not in args
    if report.get("lean_grn") is not lean:
        return f"lean GRN was {report.get('lean_grn')}, expected {lean}"
    if "--no-compile" not in args and not report.get("compiled"):
        return "compile was asked for and fell back to eager"
    return scaler_problem(report)


def scaler_problem(report: dict) -> str:
    """The trainer's own words when a group proof run's fp16 scaler collapsed;
    empty when it held or there was none."""
    if not report.get("grad_scaler"):
        return ""
    from bittrainer.generic.tasks.group_task import scaler_collapse_in

    return scaler_collapse_in(report.get("epochs") or [])


def _group(
    name: str,
    *args: str,
    minutes: float,
    trunk: Callable[[Context], str],
    against: str | None = None,
):
    """One ``group_proof`` run on this card, from the trunk *trunk* names (one
    the series made: a real run always starts from a trained model).
    ``against`` names an earlier group stage whose speed this one is compared
    with."""

    def _run(context: Context) -> Result:
        import torch

        from bittrainer.amp import native_bf16

        out = _fresh_run_folder(context, name)
        argv = _module(
            context, "bittrainer.tools.group_proof",
            "--device", "cuda", "--workdir", str(out), "--trunk", trunk(context), *args,
        )  # fmt: skip
        code, output = _command(context, name, argv, minutes)
        report = last_json(output) or {}
        numbers = group_numbers(report)
        if against:
            baseline = (
                context.report["stages"].get(against, {}).get("numbers", {})
            ).get("images_per_second")
            if baseline and numbers["images_per_second"]:
                numbers[f"speed_against_{against}"] = round(
                    numbers["images_per_second"] / baseline, 2
                )
        if code != 0 or not report.get("passed"):
            failed = [key for key, ok in (report.get("checks") or {}).items() if ok is False]
            notes = [scaler_problem(report)]
            if failed:
                notes.append(f"failed checks: {', '.join(failed)}")
            notes.append(f"exit {code}; see logs/{name}.log")
            return Result("failed", numbers, "; ".join(note for note in notes if note))
        problem = group_problem(
            report, args, native_bf16=native_bf16(torch.device("cuda"))
        )
        return Result("failed", numbers, problem) if problem else Result("passed", numbers)

    return _run


_AT_224 = ("--image-size", "224", "--batch-size", "64")

# Estimates are what each stage took on the T4 in rehearsal R1 (2026-11-10),
# rounded up; the speed stages that did not run there keep their guesses.
STAGES: list[Stage] = [
    Stage("env", "the GPU, its driver and the precision it trains in", 0.1, _env),
    Stage("gpu_tests", "the GPU-only tests: fp16 scaler, precision by card, preflight, skipped steps, lean GRN", 1.2, _gpu_tests, ("env",)),
    Stage("round_tests", "this round's test files on this machine", 3.5, _round_tests),
    Stage("learn_fp16", "a model learns in this card's precision", 0.6, _learn_fp16, ("env",)),
    Stage("convert_base", "real Apache V1 weights convert into a Base trunk", 0.2, _convert("base")),
    Stage("convert_small", "the same for the student sizes", 0.2, _convert("nano", "atto")),
    Stage("convert_tiny", "the same for tiny, the trunk the tiny group check starts from", 0.2, _convert("tiny")),
    Stage("synth_packs", "generate the packs the next stages train from", 0.2, _synth_packs),
    Stage("teacher", "train a Base teacher from a pack, Apache start", 2.6, _teacher, ("env", "synth_packs")),
    Stage("pause_resume", "pause a Base run with its PAUSE file, then resume it", 4.4, _pause_resume, ("env", "synth_packs")),
    Stage("tail", "Base at 224px and batch 64, then the 384px tail at batch 16", 1.7, _tail, ("env", "synth_packs")),
    Stage("preflight_refuses", "a 384px tail at batch 64 is refused before anything trains", 0.7, _preflight_refuses, ("env", "synth_packs")),
    Stage("distill", "distil the Base teacher into a nano", 1.8, _distill, ("teacher",)),
    Stage("distill_pause_resume", "pause and resume a distillation", 3.1, _distill_pause_resume, ("teacher",)),
    Stage("distill_channels_last", "distil with channels-last memory", 1.3, _distill_channels_last, ("teacher",)),
    Stage("soft_negatives", "train with unlabelled images as a soft no", 1.2, _soft_negatives, ("env", "synth_packs")),
    Stage("reject_negatives", "soft negatives, skipping the unlabelled pairs the model scores highest", 1.7, _unlabelled_option("reject_negatives", "--soft-negatives", "--reject-rate", "0.02"), ("env", "synth_packs")),
    Stage("hill_negatives", "the Hill term for unlabelled images", 1.7, _unlabelled_option("hill_negatives", "--hill-negatives"), ("env", "synth_packs")),
    Stage("mask_common", "soft negatives, but none for common concepts", 1.5, _unlabelled_option("mask_common", "--soft-negatives", "--mask-common", "0.05"), ("env", "synth_packs")),
    Stage("geometry", "colour-only augmentation for images with a geometry-sensitive label", 0.8, _geometry, ("env", "synth_packs")),
    Stage("auto_batch", "automatic batch size: Base with its tail, then with a teacher loaded", 6.8, _auto_batch, ("teacher",)),
    Stage("many_heads", "Base with 900 sparse heads: what the head count costs", 3.4, _many_heads, ("teacher",)),
    # The group trainers (Bitcrush ISSUE-1690): head-only, a private-names
    # scoped export and a sourceless full update, in this card's precision, on
    # generated images, each from a trained trunk the series made.
    Stage("group_base", "group, Base at 512px from the Base teacher: head-only, private export, sourceless update; compiled, lean GRN, automatic batch", 7.6, _group("group_base", "--size", "base", *_GROUP_512, minutes=7.6, trunk=_teacher_file), ("teacher",)),
    Stage("group_nano", "group, nano at 512px from the distilled nano, eager", 4, _group("group_nano", "--size", "nano", *_GROUP_SMALL, minutes=4, trunk=_distilled_nano), ("distill",)),
    Stage("group_tiny", "group, tiny at 512px from the Apache tiny, eager", 4.5, _group("group_tiny", "--size", "tiny", *_GROUP_SMALL, minutes=4.5, trunk=_converted("tiny")), ("env", "convert_tiny")),
    Stage("group_refuses", "group, Base at 512px and batch 64: refused before it trains, no backup left", 1.1, _group("group_refuses", "--size", "base", *_GROUP_512, "--images-per-class", "8", "--batch-size", "64", "--preflight", "--expect-refusal", minutes=1.1, trunk=_converted("base")), ("env", "convert_base")),
    Stage("group_pause_resume", "group, nano from the Apache nano: pause the update at step 5 (mid-epoch), resume it from its backup", 2.5, _group("group_pause_resume", "--size", "nano", *_GROUP_SMALL, "--batch-size", "16", "--epochs", "3", "--pause-after-steps", "5", minutes=2.5, trunk=_converted("nano")), ("env", "convert_small")),
    # Speed comparisons and extras: dropped first when the budget runs short.
    Stage("other_tests", "the older backbone tests on this machine", 2.5, _other_tests, optional=True),
    Stage("speed_atto", "speed: atto at 224px", 1, _speed("speed_atto", "--size", "atto", *_AT_224), ("env",), True),
    Stage("speed_nano", "speed: nano at 224px", 1.5, _speed("speed_nano", "--size", "nano", *_AT_224), ("env",), True),
    Stage("speed_tiny", "speed: tiny at 224px", 2, _speed("speed_tiny", "--size", "tiny", *_AT_224), ("env",), True),
    Stage("speed_base", "speed: base at 224px", 3, _speed("speed_base", "--size", "base", *_AT_224), ("env",), True),
    Stage("speed_base_384", "speed: base at 384px", 9, _speed("speed_base_384", "--size", "base", "--image-size", "384", "--batch-size", "16"), ("env",), True),
    Stage("speed_base_channels_last", "speed: base, channels-last", 3, _speed("speed_base_channels_last", "--size", "base", *_AT_224, "--channels-last"), ("env",), True),
    Stage("speed_base_stock_grn", "speed: base with timm's own GRN layer, batch 32", 4.5, _speed("speed_base_stock_grn", "--size", "base", "--image-size", "224", "--batch-size", "32", "--stock-grn"), ("env",), True),
    Stage("speed_base_one_loader", "speed: base, images loaded on the training thread", 4, _speed("speed_base_one_loader", "--size", "base", *_AT_224, "--workers", "0"), ("env",), True),
    Stage("group_base_eager", "group speed: Base at 512px, not compiled", 6, _group("group_base_eager", "--size", "base", *_GROUP_512, "--no-compile", minutes=6, trunk=_teacher_file, against="group_base"), ("teacher",), True),
    Stage("group_base_stock_grn", "group speed: Base at 512px, eager, timm's own GRN layer", 4.8, _group("group_base_stock_grn", "--size", "base", *_GROUP_512, "--no-compile", "--stock-grn", minutes=4.8, trunk=_teacher_file, against="group_base_eager"), ("teacher",), True),
    Stage("real_pack", "a short run on part of a real pack (needs --pack)", 6, _real_pack, ("env",), True),
]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bittrainer.tools.vm_check",
        description="Check this round's work on the rented GPU, inside a time budget.",
    )
    parser.add_argument("--work", required=True, help="folder for everything it writes")
    parser.add_argument("--budget-minutes", type=float, default=120)
    parser.add_argument(
        "--pack", default=None, help="a real pack, for the real_pack stage"
    )
    parser.add_argument("--only", default="", help="comma-separated stage names to run")
    parser.add_argument(
        "--skip", default="", help="comma-separated stage names to leave out"
    )
    parser.add_argument(
        "--redo", action="store_true", help="repeat stages that already passed"
    )
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    if args.list:
        for stage in STAGES:
            kind = "extra" if stage.optional else "must "
            print(f"{kind} {stage.minutes:>4g} min  {stage.name}: {stage.title}")
        must = sum(stage.minutes for stage in STAGES if not stage.optional)
        print(
            f"must: about {must:g} min; everything: about {sum(s.minutes for s in STAGES):g} min"
        )
        return 0

    context = Context(Path(args.work), pack=Path(args.pack) if args.pack else None)
    report = run_plan(
        STAGES,
        context,
        budget_minutes=args.budget_minutes,
        only={name for name in args.only.split(",") if name} or None,
        skip={name for name in args.skip.split(",") if name} or None,
        redo=args.redo,
    )
    lines = summary_lines(STAGES, report)
    (context.work / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    failed = [
        name
        for name, result in report["stages"].items()
        if result["status"] == "failed"
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
