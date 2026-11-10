"""The staged check for the rented GPU (Bitcrush ISSUE-1655).

``bittrainer.tools.vm_check`` runs everything this round built, on the machine it
was built for, inside a time budget: the GPU tests, fp16 learning, the real
weight conversion, training / pausing / resuming from a pack, distillation, and
speed measurements. These tests pin the parts that decide what runs and what is
reported. Nothing here trains a model or touches a GPU.
"""

from __future__ import annotations

import json
import sys
import textwrap

import bittrainer.tools.vm_check as vm
from bittrainer.backbone_pack import pack_records


def _stage(name, *, minutes=1.0, needs=(), optional=False, outcome="passed"):
    def _run(context):
        context.ran.append(name)
        return vm.Result(outcome, numbers={"stage": name})

    return vm.Stage(
        name, f"title of {name}", minutes, _run, needs=needs, optional=optional
    )


class _Clock:
    """Each stage 'takes' its estimate, so budgets are tested without waiting."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _context(tmp_path, clock=None):
    context = vm.Context(work=tmp_path, clock=clock or _Clock())
    context.ran = []
    return context


# -- the plan -----------------------------------------------------------------


def test_the_plan_fits_two_hours_with_the_musts_first():
    stages = vm.STAGES
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names))
    for stage in stages:
        assert set(stage.needs) <= set(names[: names.index(stage.name)]), stage.name
    must = sum(stage.minutes for stage in stages if not stage.optional)
    everything = sum(stage.minutes for stage in stages)
    assert must <= 60, must  # the checks that matter fit in half the budget
    assert everything <= 110, everything  # the whole plan leaves room for setup
    # Speed comparisons are the part to drop, so they come last.
    first_optional = next(i for i, stage in enumerate(stages) if stage.optional)
    assert all(stage.optional for stage in stages[first_optional:])


def test_the_plan_covers_what_was_never_run_on_a_gpu():
    names = {stage.name for stage in vm.STAGES}
    assert names >= {
        "gpu_tests",
        "learn_fp16",
        "convert_base",
        "teacher",
        "pause_resume",
        "distill",
        "distill_pause_resume",
        "auto_batch",
        "many_heads",
    }


# -- running the plan ---------------------------------------------------------


def test_stages_run_in_order_and_the_report_is_saved(tmp_path):
    context = _context(tmp_path)
    stages = [_stage("a"), _stage("b", needs=("a",))]

    report = vm.run_plan(stages, context, budget_minutes=120)

    assert context.ran == ["a", "b"]
    assert [report["stages"][name]["status"] for name in ("a", "b")] == [
        "passed",
        "passed",
    ]
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["stages"]["b"]["numbers"] == {"stage": "b"}


def test_a_failed_stage_skips_what_depends_on_it_but_not_the_rest(tmp_path):
    context = _context(tmp_path)
    stages = [
        _stage("teacher", outcome="failed"),
        _stage("distill", needs=("teacher",)),
        _stage("convert"),
    ]

    report = vm.run_plan(stages, context, budget_minutes=120)

    assert context.ran == ["teacher", "convert"]
    assert report["stages"]["distill"]["status"] == "skipped"
    assert "teacher" in report["stages"]["distill"]["note"]


def test_optional_stages_are_dropped_when_the_budget_runs_short(tmp_path):
    clock = _Clock()
    context = _context(tmp_path, clock)

    def _slow(context):
        clock.now += 50 * 60
        context.ran.append("must")
        return vm.Result("passed")

    stages = [
        vm.Stage("must", "a required check", 50, _slow),
        _stage("speed_small", minutes=5, optional=True),
        _stage("speed_big", minutes=20, optional=True),
    ]

    report = vm.run_plan(stages, context, budget_minutes=60)

    assert context.ran == ["must", "speed_small"]
    assert report["stages"]["speed_big"]["status"] == "skipped"
    assert "budget" in report["stages"]["speed_big"]["note"]


def test_nothing_starts_once_the_budget_is_spent(tmp_path):
    clock = _Clock()
    context = _context(tmp_path, clock)

    def _overruns(context):
        clock.now += 61 * 60
        return vm.Result("passed")

    stages = [vm.Stage("long", "overruns", 10, _overruns), _stage("next")]

    report = vm.run_plan(stages, context, budget_minutes=60)

    assert context.ran == []
    assert report["stages"]["next"]["status"] == "skipped"


def test_a_second_run_only_repeats_what_did_not_pass(tmp_path):
    first = _context(tmp_path)
    vm.run_plan([_stage("a"), _stage("b", outcome="failed")], first, budget_minutes=120)

    second = _context(tmp_path)
    report = vm.run_plan([_stage("a"), _stage("b")], second, budget_minutes=120)

    assert second.ran == ["b"]
    assert report["stages"]["a"]["status"] == "passed"
    assert report["stages"]["b"]["status"] == "passed"


def test_only_and_skip_choose_stages(tmp_path):
    context = _context(tmp_path)
    stages = [_stage("a"), _stage("b"), _stage("c")]

    vm.run_plan(stages, context, budget_minutes=120, only={"a", "c"}, skip={"c"})

    assert context.ran == ["a"]


def test_a_stage_that_raises_is_a_failure_not_a_crash(tmp_path):
    context = _context(tmp_path)

    def _explodes(context):
        raise RuntimeError("no GPU here")

    report = vm.run_plan(
        [vm.Stage("env", "needs a GPU", 1, _explodes), _stage("after")],
        context,
        budget_minutes=120,
    )

    assert report["stages"]["env"]["status"] == "failed"
    assert "no GPU here" in report["stages"]["env"]["note"]
    assert context.ran == ["after"]


def test_summary_lists_every_stage_with_its_numbers(tmp_path):
    context = _context(tmp_path)
    report = vm.run_plan(
        [_stage("a"), _stage("b", outcome="failed")], context, budget_minutes=120
    )

    text = "\n".join(vm.summary_lines([_stage("a"), _stage("b")], report))

    assert "passed" in text and "FAILED" in text
    assert "title of a" in text and "stage=a" in text


# -- the generated pack -------------------------------------------------------


def test_synthetic_pack_is_a_readable_pack(tmp_path):
    pack = vm.write_synthetic_pack(tmp_path / "pack", images=24, image_size=48)

    records, counts = pack_records(pack)

    assert counts == {"images": 24, "missing_files": 0}
    heads = {head for record in records for head in record["binary"]}
    assert heads == {"c1", "c2"}
    assert {key for record in records for key in record["groups"]} == {"g1"}


def test_synthetic_pack_can_carry_hundreds_of_sparse_heads(tmp_path):
    pack = vm.write_synthetic_pack(
        tmp_path / "pack", images=60, image_size=48, extra_heads=200
    )

    records, _ = pack_records(pack)

    heads = {head for record in records for head in record["binary"]}
    assert len(heads) > 150  # sparse labels: most extra heads land on some image
    per_image = sum(len(record["binary"]) for record in records) / len(records)
    assert per_image < 40  # an image carries a few dozen answers, not all 200
    answers = {a for record in records for a in record["binary"].values()}
    assert answers == {"positive", "negative"}


# -- reading results ----------------------------------------------------------


def test_the_proof_report_is_read_out_of_its_output():
    output = textwrap.dedent(
        """\
        some warning line
        {
          "amp_dtype": "float16",
          "passed": true,
          "epochs": [{"epoch": 1}]
        }
          completed: ok
        PASS
        """
    )
    assert vm.last_json(output) == {
        "amp_dtype": "float16",
        "passed": True,
        "epochs": [{"epoch": 1}],
    }
    assert vm.last_json("no report here") is None


def test_run_numbers_come_from_the_progress_file(tmp_path):
    frames = [
        {"stage": "preparing", "train_samples": 10, "amp_dtype": "float16", "grad_scaler": True},
        {"stage": "training_step", "images_per_second": 99.0},
        {"stage": "training", "loss": 0.7, "epoch": 1, "images_per_second": 50.0},
        {"type": "autobatch", "batch_size": 96},
        {"type": "training_resumed", "epoch": 1},
        {"stage": "training", "loss": 0.6, "epoch": 2, "images_per_second": 55.0,
         "loss_terms": {"head": 0.2}},
    ]  # fmt: skip
    (tmp_path / "progress.jsonl").write_text(
        "\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8"
    )

    numbers = vm.run_numbers(tmp_path)

    assert numbers["images_per_second"] == 55.0
    assert numbers["precision"] == "float16" and numbers["grad_scaler"] is True
    assert numbers["epochs"] == 2 and numbers["resumed"] is True
    assert numbers["auto_batch_size"] == 96
    assert numbers["loss_terms"] == {"head": 0.2}


# -- pausing and resuming a detached run --------------------------------------


_FAKE_RUNNER = """
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
progress = out / "progress.jsonl"
resumed = progress.exists()
with progress.open("a", encoding="utf-8") as handle:
    if resumed:
        handle.write(json.dumps({"type": "training_resumed", "epoch": 0}) + "\\n")
        sys.exit(0)
    handle.write(json.dumps({"stage": "training_step", "batch": 1}) + "\\n")
    handle.flush()
    for _ in range(200):
        if (out / "PAUSE").exists():
            (out / "PAUSE").unlink()
            sys.exit(3)
        time.sleep(0.05)
sys.exit(9)
"""


def test_pause_then_resume_drives_a_run_through_its_control_file(tmp_path):
    script = tmp_path / "fake_runner.py"
    script.write_text(_FAKE_RUNNER, encoding="utf-8")
    out = tmp_path / "run"
    context = _context(tmp_path)

    result = vm.pause_then_resume(
        context, "fake", [sys.executable, str(script), str(out)], out, poll_seconds=0.05
    )

    assert result.status == "passed", result.note
    assert result.numbers["resumed"] is True


def test_a_run_that_never_pauses_fails_the_check(tmp_path):
    script = tmp_path / "ignores_pause.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'progress.jsonl').write_text("
        "json.dumps({'stage': 'training_step'}) + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"

    result = vm.pause_then_resume(
        _context(tmp_path), "fake", [sys.executable, str(script), str(out)], out,
        poll_seconds=0.05,
    )  # fmt: skip

    assert result.status == "failed"
    assert "exit" in result.note


def test_base_stages_ask_for_what_a_16gb_card_holds():
    """First run on the T4: Base ran out of memory at batch 128, and at 384px
    with batch 32. The comparison that replaced them measures what the lean GRN
    layer buys, so that gain stays under watch."""
    names = {stage.name for stage in vm.STAGES}

    assert "speed_base_batch128" not in names
    assert "speed_base_stock_grn" in names
