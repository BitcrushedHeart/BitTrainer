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


def test_run_numbers_show_the_sizes_the_preflight_and_the_skipped_steps(tmp_path):
    frames = [
        {"stage": "preflight", "phase": "main", "peak_vram_mb": 11400},
        {"stage": "preflight", "phase": "tail", "peak_vram_mb": 9400},
        {"type": "autobatch", "phase": "main", "batch_size": 72},
        {"type": "autobatch", "phase": "tail", "batch_size": 21},
        {"stage": "training", "loss": 0.7, "epoch": 1, "image_size": 224, "batch_size": 64},
        {"stage": "warning", "warning": "skipped_steps"},
        {"stage": "training", "loss": 0.6, "epoch": 2, "image_size": 384, "batch_size": 16},
        {"stage": "training", "loss": 0.5, "epoch": 3, "image_size": 384, "batch_size": 16},
    ]  # fmt: skip
    (tmp_path / "progress.jsonl").write_text(
        "\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8"
    )
    (tmp_path / "result.json").write_text(
        json.dumps({"validation_score": 0.8, "skipped_steps": 16, "grad_scale": 1.0}),
        encoding="utf-8",
    )

    numbers = vm.run_numbers(tmp_path)

    assert numbers["phases"] == ["224px x64", "384px x16"]
    assert numbers["preflight_mb"] == {"main": 11400, "tail": 9400}
    assert numbers["auto_batch_size"] == 72 and numbers["auto_tail_batch_size"] == 21
    assert numbers["warnings"] == 1
    assert numbers["skipped_steps"] == 16 and numbers["grad_scale"] == 1.0


# -- the tail's batch size, the preflight and automatic sizing -----------------


def test_the_plan_proves_the_tail_batch_size_and_the_refusal():
    by_name = {stage.name: stage for stage in vm.STAGES}
    assert not by_name["tail"].optional
    assert not by_name["preflight_refuses"].optional
    # This round's new test files run on the rented GPU too, where their GPU
    # tests do not skip.
    assert {
        "test_backbone_tail_batch.py",
        "test_backbone_skipped_steps.py",
        "test_backbone_auto_batch.py",
        "test_backbone_run_record.py",
    } <= set(vm._ROUND_TESTS)
    assert set(vm._GPU_TESTS) <= set(vm._ROUND_TESTS)
    assert all((vm._TESTS / name).is_file() for name in vm._ROUND_TESTS)


def _commands(monkeypatch, exit_code=0, output=""):
    """The command lines the stages would run, without running them."""
    ran: list[list[str]] = []

    def _command(context, name, argv, minutes):
        ran.append(argv)
        return exit_code, output

    monkeypatch.setattr(vm, "_command", _command)
    return ran


def test_the_tail_stage_runs_base_at_64_then_the_tail_at_16(tmp_path, monkeypatch):
    ran = _commands(monkeypatch)

    vm._tail(_context(tmp_path))

    (argv,) = ran
    value = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
    assert (value("--size"), value("--init")) == ("base", "apache")
    assert (value("--batch-size"), value("--finetune-batch-size")) == ("64", "16")
    assert value("--finetune-image-size") == "384"
    assert "--image-size" not in argv  # the runner's own default, 224


def test_a_tail_run_passes_only_if_both_sizes_trained_and_were_checked_first():
    good = {
        "phases": ["224px x64", "384px x16"],
        "preflight_mb": {"main": 11000, "tail": 9500},
    }
    assert vm.tail_problem(good) == ""
    same_batch = {**good, "phases": ["224px x64", "384px x64"]}
    assert "384px x64" in vm.tail_problem(same_batch)
    assert "preflight" in vm.tail_problem({**good, "preflight_mb": {"main": 11000}})
    assert vm.tail_problem({}) != ""


def test_a_refused_run_must_exit_2_name_the_tail_and_train_nothing():
    said = (
        "[train] REFUSED: Preflight: one training step of the tail (384px, batch "
        "size 64, precision float16) ran out of GPU memory."
    )
    assert vm.refusal_problem(2, said, '{"stage": "preflight"}') == ""
    assert "exit 1" in vm.refusal_problem(1, said, "")  # failed, not refused
    assert "exit 0" in vm.refusal_problem(0, said, "")  # it went ahead
    assert vm.refusal_problem(2, "[train] REFUSED: the pack has no images", "") != ""
    assert "trained" in vm.refusal_problem(2, said, '{"stage": "training_step"}')


def test_the_refusal_stage_asks_for_a_tail_that_cannot_fit(tmp_path, monkeypatch):
    said = "[train] REFUSED: Preflight: one training step of the tail ran out"
    ran = _commands(monkeypatch, exit_code=2, output=said)

    result = vm._preflight_refuses(_context(tmp_path))

    (argv,) = ran
    assert argv[argv.index("--finetune-batch-size") + 1] == "64"
    assert argv[argv.index("--finetune-image-size") + 1] == "384"
    assert result.status == "passed", result.note

    _commands(monkeypatch, exit_code=0, output="[train] finished")
    assert vm._preflight_refuses(_context(tmp_path)).status == "failed"


def test_automatic_sizes_must_be_useful_not_just_survivable():
    good = {"auto_batch_base": 72, "auto_batch_base_tail": 21}
    assert vm.auto_batch_problem(good) == ""
    # What the first version chose: it ran, and left most of the card unused.
    assert "8" in vm.auto_batch_problem({**good, "auto_batch_base": 8})
    assert vm.auto_batch_problem({**good, "auto_batch_base_tail": None}) != ""
    assert vm.auto_batch_problem({**good, "auto_batch_base_tail": 72}) != ""


def test_the_gpu_tests_stage_fails_when_a_gpu_test_is_skipped(tmp_path, monkeypatch):
    context = _context(tmp_path)
    _commands(monkeypatch, output="===== 7 passed, 80 deselected in 20.1s =====")
    assert vm._gpu_tests(context).status == "passed"
    _commands(monkeypatch, output="===== 5 passed, 2 skipped, 80 deselected =====")
    assert vm._gpu_tests(context).status == "failed"
    _commands(monkeypatch, output="===== 7 skipped, 80 deselected in 2.0s =====")
    assert vm._gpu_tests(context).status == "failed"


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


# -- the group trainers (Bitcrush ISSUE-1690) -----------------------------------


def _value(argv, flag):
    return argv[argv.index(flag) + 1]


def _by_name():
    return {stage.name: stage for stage in vm.STAGES}


def test_the_plan_covers_the_group_trainers_on_the_t4():
    stages = _by_name()
    for name in ("group_base", "group_nano", "group_tiny", "group_refuses", "group_pause_resume"):
        assert not stages[name].optional, name
    for name in ("group_base_eager", "group_base_stock_grn"):
        assert stages[name].optional, name
    group_tests = {
        "test_group_amp.py",
        "test_group_lean_grn.py",
        "test_group_preflight.py",
        "test_group_proof.py",
    }
    assert group_tests <= set(vm._ROUND_TESTS)
    assert group_tests - {"test_group_proof.py"} <= set(vm._GPU_TESTS)


def test_the_group_stages_ask_for_what_they_prove(tmp_path, monkeypatch):
    ran = _commands(monkeypatch)
    context = _context(tmp_path)
    names = (
        "group_base", "group_nano", "group_tiny", "group_refuses",
        "group_pause_resume", "group_base_eager", "group_base_stock_grn",
    )  # fmt: skip
    for name in names:
        assert _by_name()[name].run(context).status == "failed"  # no report: no pass
    argvs = dict(zip(names, ran, strict=True))

    for name, argv in argvs.items():
        assert argv[1:3] == ["-m", "bittrainer.tools.group_proof"], name
        assert _value(argv, "--device") == "cuda" and _value(argv, "--bucket-size") == "512"
        assert _value(argv, "--workdir").endswith(name)
    sizes = {name: _value(argv, "--size") for name, argv in argvs.items()}
    assert sizes["group_base"] == sizes["group_refuses"] == "base"
    assert (sizes["group_nano"], sizes["group_tiny"]) == ("nano", "tiny")
    # Base trains as Engine runs it: compiled, lean GRN, automatic batch size.
    base = argvs["group_base"]
    assert "--no-compile" not in base and "--stock-grn" not in base
    assert "--batch-size" not in base
    assert "--no-compile" in argvs["group_base_eager"]
    assert "--stock-grn" in argvs["group_base_stock_grn"]
    refuse = argvs["group_refuses"]
    assert "--expect-refusal" in refuse and "--preflight" in refuse
    assert _value(refuse, "--batch-size") == "64"
    pause = argvs["group_pause_resume"]
    assert _value(pause, "--pause-after-steps") == "5"
    # Mid-epoch: more steps in an epoch than the pause waits for.
    assert int(_value(pause, "--epochs")) >= 2


def test_a_group_run_must_have_trained_the_way_this_card_trains():
    t4 = {"amp_dtype": "float16", "grad_scaler": True, "lean_grn": True, "compiled": True}
    assert vm.group_problem(t4, ("--size", "base"), native_bf16=False) == ""
    assert "float16" in vm.group_problem({**t4, "amp_dtype": "bfloat16"}, (), native_bf16=False)
    assert "GradScaler" in vm.group_problem({**t4, "grad_scaler": False}, (), native_bf16=False)
    assert "lean GRN" in vm.group_problem({**t4, "lean_grn": False}, (), native_bf16=False)
    assert vm.group_problem({**t4, "lean_grn": False}, ("--stock-grn",), native_bf16=False) == ""
    assert "eager" in vm.group_problem({**t4, "compiled": False}, (), native_bf16=False)
    assert vm.group_problem({**t4, "compiled": False}, ("--no-compile",), native_bf16=False) == ""
    ampere = {"amp_dtype": "bfloat16", "grad_scaler": False, "lean_grn": False, "compiled": True}
    assert vm.group_problem(ampere, (), native_bf16=True) == ""
    assert vm.group_problem({}, ("--expect-refusal",), native_bf16=False) == ""


def test_a_group_stage_reports_its_speed_and_passes_on_a_passing_proof(
    tmp_path, monkeypatch
):
    import bittrainer.amp as amp

    monkeypatch.setattr(amp, "native_bf16", lambda device: False)  # a T4
    report = {
        "passed": True, "amp_dtype": "float16", "grad_scaler": True, "lean_grn": True,
        "compiled": True, "best_images_per_second": 12.04, "batch_size_used": 8,
        "peak_vram_mb": 11800, "preflight_mb": 11200, "skipped_steps": 14,
    }  # fmt: skip
    _commands(monkeypatch, output=json.dumps(report, indent=2))
    context = _context(tmp_path)

    base = _by_name()["group_base"].run(context)

    assert base.status == "passed", base.note
    assert base.numbers["images_per_second"] == 12.0
    assert base.numbers["precision"] == "float16" and base.numbers["grad_scaler"] is True
    assert base.numbers["batch_size"] == 8 and base.numbers["skipped_steps"] == 14

    # The eager run is held against the compiled one's speed.
    context.report["stages"]["group_base"] = {"numbers": base.numbers}
    eager_report = {**report, "compiled": False, "best_images_per_second": 9.0}
    _commands(monkeypatch, output=json.dumps(eager_report, indent=2))
    eager = _by_name()["group_base_eager"].run(context)
    assert eager.status == "passed", eager.note
    assert eager.numbers["speed_against_group_base"] == 0.75

    # A proof that trained in the wrong precision fails the stage, as does one
    # that failed its own checks.
    _commands(monkeypatch, output=json.dumps({**report, "grad_scaler": False}, indent=2))
    assert "GradScaler" in _by_name()["group_nano"].run(context).note
    _commands(monkeypatch, exit_code=1, output=json.dumps({**report, "passed": False}, indent=2))
    assert _by_name()["group_tiny"].run(context).status == "failed"


# -- a group stage starts like a real run and notices a collapsed scaler --------
# (Bitcrush ISSUE-1690). On the T4 on 2026-11-10 the group stages started from
# random weights: the fp16 scaler fell to 0.0625 with most of the first epoch's
# steps skipped, and every stage passed. From the series' Base teacher: 2
# skipped steps, scale 16384, loss falling.

_TEACHER = "runs/teacher/backbone_candidate.safetensors"
_TRUNK_OF = {
    "group_base": ("teacher", _TEACHER),
    "group_nano": ("distill", "runs/distill/backbone_candidate.safetensors"),
    "group_tiny": ("convert_tiny", "trunks/tiny.safetensors"),
    "group_refuses": ("convert_base", "trunks/base.safetensors"),
    "group_pause_resume": ("convert_small", "trunks/nano.safetensors"),
    "group_base_eager": ("teacher", _TEACHER),
    "group_base_stock_grn": ("teacher", _TEACHER),
}


def test_every_group_stage_starts_from_a_trunk_the_series_made(tmp_path, monkeypatch):
    ran = _commands(monkeypatch)
    context = _context(tmp_path)
    stages = _by_name()
    names = [name for name in stages if name.startswith("group_")]
    assert set(names) == set(_TRUNK_OF)
    for name in names:
        stages[name].run(context)

    made_by = {
        "teacher": _value(list(vm._BASE), "--size"),
        "distill": _value(list(vm._NANO), "--size"),
    }
    for name, argv in zip(names, ran, strict=True):
        maker, trunk = _TRUNK_OF[name]
        assert _value(argv, "--trunk") == str(tmp_path / trunk), name
        assert maker in stages[name].needs, name
        # The trunk is the size the stage trains, and never timm's own weights
        # (their licence is non-commercial).
        trunk_size = made_by.get(maker) or trunk.split("/")[-1].split(".")[0]
        assert trunk_size == _value(argv, "--size"), name
        assert "timm" not in argv, name


def test_the_tiny_trunk_is_converted_like_the_others(tmp_path, monkeypatch):
    ran = _commands(monkeypatch)
    stage = _by_name()["convert_tiny"]

    assert stage.run(_context(tmp_path)).status == "passed"

    (argv,) = ran
    assert argv[1:3] == ["-m", "bittrainer.tools.convnext_v1_trunk"]
    assert _value(argv, "--size") == "tiny"
    assert _value(argv, "--out") == str(tmp_path / "trunks" / "tiny.safetensors")
    assert not stage.optional


_T4 = {"amp_dtype": "float16", "grad_scaler": True, "lean_grn": True, "compiled": True}


def _epochs(*rows):
    keys = ("epoch", "steps", "skipped_steps", "late_skipped_steps", "grad_scale")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def test_a_group_stage_whose_scaler_collapsed_fails_and_says_why():
    # From the teacher on 2026-11-10: two skips in the opening descent.
    healthy = {**_T4, "grad_scale": 16384.0, "epochs": _epochs(
        (1, 11, 2, 0, 16384.0), (2, 11, 0, 0, 16384.0), (3, 11, 0, 0, 16384.0))}  # fmt: skip
    assert vm.group_problem(healthy, (), native_bf16=False) == ""

    # From random weights: the first epoch all skipped, the scale under 1.
    random_start = {**_T4, "grad_scale": 0.0625, "epochs": _epochs(
        (1, 11, 11, 0, 32.0), (2, 11, 9, 9, 0.0625))}  # fmt: skip
    problem = vm.group_problem(random_start, (), native_bf16=False)
    assert "GradScaler collapsed" in problem and "below 1" in problem

    # Still above 1, but most of an epoch skipped after the scaler settled.
    stalled = {**_T4, "grad_scale": 64.0, "epochs": _epochs(
        (1, 11, 4, 0, 4096.0), (2, 10, 6, 6, 64.0))}  # fmt: skip
    problem = vm.group_problem(stalled, (), native_bf16=False)
    assert "GradScaler collapsed" in problem and "6 of its 10 steps" in problem

    # An opening descent that fills a short first epoch is the scaler finding
    # its level, not a collapse.
    opening = {**_T4, "grad_scale": 4096.0, "epochs": _epochs(
        (1, 4, 4, 0, 4096.0), (2, 4, 0, 0, 4096.0))}  # fmt: skip
    assert vm.group_problem(opening, (), native_bf16=False) == ""

    # A bf16 card has no scaler; a refusal trained nothing.
    ampere = {**random_start, "amp_dtype": "bfloat16", "grad_scaler": False, "lean_grn": False}
    assert vm.group_problem(ampere, (), native_bf16=True) == ""
    assert vm.group_problem(random_start, ("--expect-refusal",), native_bf16=False) == ""


def test_a_group_stage_reports_the_collapse_even_when_the_proof_failed_on_it(
    tmp_path, monkeypatch
):
    import bittrainer.amp as amp

    monkeypatch.setattr(amp, "native_bf16", lambda device: False)  # a T4
    collapsed = {
        **_T4, "passed": True, "grad_scale": 0.0625,
        "epochs": _epochs((1, 11, 11, 0, 32.0), (2, 11, 9, 9, 0.0625)),
    }  # fmt: skip
    _commands(monkeypatch, output=json.dumps(collapsed, indent=2))
    result = _by_name()["group_base"].run(_context(tmp_path))
    assert result.status == "failed" and "GradScaler collapsed" in result.note
    assert result.numbers["grad_scale"] == 0.0625

    failed = {**collapsed, "passed": False, "checks": {"scaler_held": False}}
    _commands(monkeypatch, exit_code=1, output=json.dumps(failed, indent=2))
    result = _by_name()["group_nano"].run(_context(tmp_path))
    assert result.status == "failed"
    assert "GradScaler collapsed" in result.note and "scaler_held" in result.note


# What each stage took on the T4 in rehearsal R1 (2026-11-10), in seconds. The
# group stages then started from random weights; from a trunk they train the
# same steps on the same images.
_R1_SECONDS = {
    "gpu_tests": 71.7, "round_tests": 154.4, "learn_fp16": 31.3, "convert_base": 10.5,
    "convert_small": 11.1, "synth_packs": 6.7, "teacher": 154.9, "pause_resume": 260.8,
    "tail": 98.0, "preflight_refuses": 40.6, "distill": 106.8,
    "distill_pause_resume": 181.2, "distill_channels_last": 74.2, "soft_negatives": 68.9,
    "reject_negatives": 98.1, "hill_negatives": 97.7, "mask_common": 89.0, "geometry": 45.0,
    "auto_batch": 407.0, "many_heads": 202.2, "group_base": 449.5, "group_nano": 232.6,
    "group_tiny": 263.7, "group_refuses": 61.6, "group_pause_resume": 147.1,
    "other_tests": 146.2, "group_base_eager": 349.8, "group_base_stock_grn": 283.3,
}  # fmt: skip


def test_no_estimate_is_below_what_the_t4_measured():
    stages = _by_name()
    for name, seconds in _R1_SECONDS.items():
        assert stages[name].minutes * 60 >= seconds, name


def test_the_private_export_tests_run_on_the_rented_gpu_too():
    assert "test_cloud_export_private_names.py" in vm._ROUND_TESTS


def test_base_stages_ask_for_what_a_16gb_card_holds():
    """First run on the T4: Base ran out of memory at batch 128, and at 384px
    with batch 32. The comparison that replaced them measures what the lean GRN
    layer buys, so that gain stays under watch."""
    names = {stage.name for stage in vm.STAGES}

    assert "speed_base_batch128" not in names
    assert "speed_base_stock_grn" in names
