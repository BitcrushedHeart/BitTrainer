"""Proof run for the Backbone Builder trainer on generated images (Bitcrush ISSUE-1655).

``bittrainer.tools.backbone_proof`` draws its own labelled images, trains a
backbone on them end to end and reports whether the run held together and how
fast it went. The same command runs at home and on a rented GPU, so no real
image ever has to be read to prove the trainer.

CPU-only, ``atto`` backbone, a couple of dozen 64px images.
"""

from __future__ import annotations

from pathlib import Path

import bittrainer.tools.backbone_proof as proof


def test_fixture_dataset_is_deterministic_and_labelled(tmp_path):
    a = proof.make_fixture_dataset(tmp_path / "a", count=36, image_size=48, seed=3)
    b = proof.make_fixture_dataset(tmp_path / "b", count=36, image_size=48, seed=3)

    assert len(a) == 36
    labels = lambda records: [  # noqa: E731
        (r["content_hash"], r["binary"], r["groups"]) for r in records
    ]
    assert labels(a) == labels(b)
    first_a, first_b = Path(a[0]["file_paths"][0]), Path(b[0]["file_paths"][0])
    assert first_a.is_file() and first_a.read_bytes() == first_b.read_bytes()
    assert len({r["content_hash"] for r in a}) == 36

    # Every head has both answers, and some labels are withheld so the masked
    # loss is exercised the way a real, partly-labelled pool exercises it.
    for concept in proof.CONCEPTS:
        seen = [r["binary"].get(concept) for r in a]
        assert "positive" in seen and "negative" in seen and None in seen
    classes = [r["groups"].get(proof.GROUP) for r in a]
    assert set(proof.LAYOUTS) <= set(classes) and None in classes
    assert all(r["binary"] or r["groups"] for r in a)


def test_a_different_seed_draws_different_images(tmp_path):
    a = proof.make_fixture_dataset(tmp_path / "a", count=12, image_size=48, seed=1)
    b = proof.make_fixture_dataset(tmp_path / "b", count=12, image_size=48, seed=2)
    assert (
        Path(a[0]["file_paths"][0]).read_bytes()
        != Path(b[0]["file_paths"][0]).read_bytes()
    )


def test_precision_choice_maps_onto_the_training_config(tmp_path):
    records = proof.make_fixture_dataset(tmp_path, count=12, image_size=48)

    def config(**kw):
        return proof.build_request(tmp_path, records, **kw)["training_config"]

    assert config(amp="off")["use_amp"] is False
    assert config(amp="fp16")["use_amp"] is True
    assert config(amp="fp16")["amp_dtype"] == "float16"
    assert config(amp="bf16")["amp_dtype"] == "bfloat16"
    assert config(amp="auto")["amp_dtype"] == "auto"
    assert config(workers=4, channels_last=True)["dataloader_workers"] == 4
    assert config(workers=4, channels_last=True)["channels_last"] is True

    request = proof.build_request(tmp_path, records, size="base", init="timm")
    assert request["convnextv2_size"] == "base"
    assert request["backbone_init"]["source"] == "timm_pretrained"
    assert (
        proof.build_request(tmp_path, records)["backbone_init"]["source"]
        == "random_init"
    )


def test_proof_run_completes_on_cpu(tmp_path):
    report = proof.run_proof(
        tmp_path,
        images=24,
        image_size=64,
        size="atto",
        epochs=1,
        batch_size=8,
        device="cpu",
        amp="off",
        min_score=None,
    )

    checks = report["checks"]
    assert checks["completed"] is True
    assert checks["loss_finite"] is True
    assert checks["checkpoint_finite"] is True
    assert checks["loss_fell"] is None  # one epoch: nothing to compare against
    assert checks["score_reached"] is None  # no floor was asked for
    assert report["passed"] is True

    assert report["device"] == "cpu"
    assert report["amp_dtype"] == "off"
    assert report["grad_scaler"] is False
    assert len(report["epochs"]) == 1
    assert report["epochs"][0]["images_per_second"] > 0
    assert report["best_images_per_second"] > 0
    # Generated images only: the report names no file.
    assert "records" not in report and "file_paths" not in str(report)


def test_a_failed_check_fails_the_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        proof,
        "run_proof",
        lambda root, **kw: {
            "passed": False,
            "checks": {"completed": True, "score_reached": False},
        },
    )
    assert proof.main(["--workdir", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().out

    monkeypatch.setattr(
        proof,
        "run_proof",
        lambda root, **kw: {"passed": True, "checks": {"completed": True}},
    )
    assert proof.main(["--workdir", str(tmp_path)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_command_line_options_reach_the_run(monkeypatch, tmp_path):
    seen: dict = {}

    def _fake(root, **kw):
        seen.update(kw, root=Path(root))
        return {"passed": True, "checks": {}}

    monkeypatch.setattr(proof, "run_proof", _fake)
    code = proof.main(
        [
            "--workdir", str(tmp_path),
            "--size", "base",
            "--image-size", "224",
            "--images", "4000",
            "--epochs", "2",
            "--batch-size", "64",
            "--workers", "4",
            "--amp", "fp16",
            "--channels-last",
            "--init", "timm",
            "--min-score", "0.7",
            "--device", "cuda",
            "--seed", "5",
        ]
    )  # fmt: skip

    assert code == 0
    assert seen == {
        "root": tmp_path,
        "size": "base",
        "image_size": 224,
        "images": 4000,
        "epochs": 2,
        "batch_size": 64,
        "workers": 4,
        "amp": "fp16",
        "channels_last": True,
        "stock_grn": False,
        "init": "timm",
        "min_score": 0.7,
        "device": "cuda",
        "seed": 5,
    }


def test_the_stock_grn_layer_can_be_asked_for(monkeypatch, tmp_path):
    records = proof.make_fixture_dataset(tmp_path, count=12, image_size=48)

    default = proof.build_request(tmp_path, records)["training_config"]
    stock = proof.build_request(tmp_path, records, stock_grn=True)["training_config"]
    # Unset means the trainer's own default, which is the lean layer.
    assert "lean_grn" not in default
    assert stock["lean_grn"] is False

    seen: dict = {}

    def _fake(root, **kw):
        seen.update(kw)
        return {"passed": True, "checks": {}}

    monkeypatch.setattr(proof, "run_proof", _fake)
    proof.main(["--workdir", str(tmp_path), "--stock-grn"])
    assert seen["stock_grn"] is True
