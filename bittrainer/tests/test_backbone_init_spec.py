from __future__ import annotations

import ast
from pathlib import Path

import torch


class TinyModel(torch.nn.Module):
    def __init__(self, *, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, num_classes)
        self.num_features = 1

    def get_classifier(self):
        return self.head

    def forward(self, x):  # pragma: no cover - not used by these tests
        return self.head(self.stem(x))


def test_create_model_uses_local_safetensors_without_timm_pretrained(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    import bittrainer.model as model_mod

    checkpoint = tmp_path / "active.safetensors"
    expected_weight = torch.full((1, 1), 0.75)
    save_file(
        {"stem.weight": expected_weight, "stem.bias": torch.full((1,), 0.25)},
        str(checkpoint),
        metadata={"source": "test"},
    )
    calls: list[dict] = []

    def fake_timm_create_model(model_name, *, pretrained, num_classes, head_hidden_size=None):
        calls.append(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "num_classes": num_classes,
                "head_hidden_size": head_hidden_size,
            }
        )
        return TinyModel(num_classes=num_classes)

    monkeypatch.setattr(model_mod.timm, "create_model", fake_timm_create_model)

    created = model_mod.create_model(
        model_size="nano",
        pretrained=True,
        num_classes=2,
        backbone_init={
            "source": "local_active",
            "checkpoint_path": str(checkpoint),
        },
    )

    assert calls[0]["pretrained"] is False
    assert torch.equal(created.stem.weight.detach(), expected_weight)


def test_create_model_preserves_temporary_timm_fallback(monkeypatch):
    import bittrainer.model as model_mod

    calls: list[dict] = []

    def fake_timm_create_model(model_name, *, pretrained, num_classes, head_hidden_size=None):
        calls.append({"pretrained": pretrained, "num_classes": num_classes})
        return TinyModel(num_classes=num_classes)

    monkeypatch.setattr(model_mod.timm, "create_model", fake_timm_create_model)

    model_mod.create_model(
        model_size="nano",
        pretrained=False,
        num_classes=2,
        backbone_init={"source": "temporary_timm_pretrained_fallback"},
    )

    assert calls == [{"pretrained": True, "num_classes": 2}]


def test_create_model_supports_explicit_random_init(monkeypatch):
    import bittrainer.model as model_mod

    calls: list[dict] = []

    def fake_timm_create_model(model_name, *, pretrained, num_classes, head_hidden_size=None):
        calls.append({"pretrained": pretrained, "num_classes": num_classes})
        return TinyModel(num_classes=num_classes)

    monkeypatch.setattr(model_mod.timm, "create_model", fake_timm_create_model)

    model_mod.create_model(
        model_size="nano",
        pretrained=True,
        num_classes=2,
        backbone_init={"source": "random_init"},
    )

    assert calls == [{"pretrained": False, "num_classes": 2}]


def test_group_warmstart_passes_backbone_init_to_model_factory(monkeypatch, tmp_path):
    import bittrainer.group_trainer as group_trainer

    calls: list[dict] = []
    spec = {"source": "local_active", "checkpoint_path": "C:/backbones/active.safetensors"}

    def fake_create_model(**kwargs):
        calls.append(kwargs)
        return TinyModel(num_classes=kwargs["num_classes"])

    monkeypatch.setattr(group_trainer, "create_model", fake_create_model)

    config = group_trainer.GroupTrainConfig(
        group_folder=str(tmp_path),
        num_classes=3,
        class_names=["a", "b", "c"],
        backbone_variant="nano",
        backbone_init=spec,
        from_scratch=False,
    )

    group_trainer._create_or_warmstart_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=None,
        checkpoint_dir=tmp_path,
    )

    assert calls[0]["pretrained"] is True
    assert calls[0]["backbone_init"] == spec


def test_multihead_and_dual_branch_models_forward_backbone_init(monkeypatch):
    import bittrainer.dual_branch_model as dual_branch_model
    import bittrainer.multihead_model as multihead_model

    calls: list[dict] = []
    spec = {"source": "random_init"}

    def fake_create_model(**kwargs):
        calls.append(kwargs)
        return TinyModel(num_classes=kwargs["num_classes"])

    monkeypatch.setattr(multihead_model, "create_model", fake_create_model)
    monkeypatch.setattr(dual_branch_model, "create_model", fake_create_model)

    multihead_model.MultiHeadConvNeXt(
        backbone_variant="nano",
        n_bands=2,
        n_sizes=3,
        backbone_init=spec,
    )
    dual_branch_model.DualBranchConvNeXt(
        backbone_variant="nano",
        num_classes=2,
        backbone_init=spec,
    )

    assert calls[0]["backbone_init"] == spec
    assert calls[0]["pretrained"] is True
    assert calls[1]["backbone_init"] == spec
    assert calls[2]["backbone_init"] == spec


def test_trainers_do_not_call_timm_create_model_directly():
    root = Path(__file__).resolve().parents[2]
    allowed = {root / "bittrainer" / "model.py"}
    offenders: list[str] = []
    for path in (root / "bittrainer").glob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "create_model"
                and isinstance(func.value, ast.Name)
                and func.value.id == "timm"
            ):
                offenders.append(str(path.relative_to(root)))

    assert offenders == []
