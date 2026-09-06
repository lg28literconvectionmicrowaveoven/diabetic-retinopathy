"""Ensemble CAM equivalence and checkpoint-discovery tests.

Run with:  python -m pytest tests/test_ensemble.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

sys.modules["torchaudio"] = None

import torch
from transformers.models.siglip.modeling_siglip import SiglipVisionConfig, SiglipVisionModel

from explainability import MedSigLIPExplainableModel
from models import MLPHead
from utils import discover_fold_checkpoints


def build_model(seed: int = 0):
    torch.manual_seed(seed)
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=56,
        patch_size=14,
    )
    vision = SiglipVisionModel(config)
    head = MLPHead(in_dim=64, hidden_dim=32, out_dim=5, dropout=0.0)
    head.eval()
    return vision, head


def test_single_head_ensemble_matches_forward_with_explainability():
    vision, head = build_model(0)
    for p in vision.parameters():
        p.requires_grad = False
    vision.eval()

    pixel_values = torch.randn(1, 3, 56, 56)

    with MedSigLIPExplainableModel(
        encoder=vision, classifier=head, grid_size=4, embed_dim=64
    ) as model:
        model.eval()
        _, explanation = model.forward_with_explainability(
            pixel_values, target_size=(56, 56), num_classes=5
        )
        probs, cam = model.forward_ensemble_cams(
            pixel_values, [head], num_classes=5, target_size=(56, 56)
        )

        assert torch.allclose(probs, torch.softmax(explanation.logits, dim=-1), atol=1e-6)
        assert torch.allclose(cam, explanation.predicted_attribution, atol=1e-5), (
            "ensemble CAM path must reproduce the single-head CAM exactly"
        )


def test_multi_head_ensemble_probabilities_and_cam():
    vision, head_a = build_model(1)
    _, head_b = build_model(2)
    for p in vision.parameters():
        p.requires_grad = False
    vision.eval()

    pixel_values = torch.randn(1, 3, 56, 56)

    with MedSigLIPExplainableModel(
        encoder=vision, classifier=head_a, grid_size=4, embed_dim=64
    ) as model:
        model.eval()
        probs, cam = model.forward_ensemble_cams(
            pixel_values, [head_a, head_b], num_classes=5, target_size=(56, 56)
        )

        with torch.no_grad():
            expected = (
                torch.softmax(head_a(vision(pixel_values).pooler_output), dim=-1)
                + torch.softmax(head_b(vision(pixel_values).pooler_output), dim=-1)
            ) / 2

        assert torch.allclose(probs, expected, atol=1e-6)
        assert cam.shape == (1, 56, 56)
        assert 0.0 <= cam.min().item() and cam.max().item() <= 1.0


def test_discover_fold_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    base = checkpoint_dir / "multiclass"
    for i in (4, 0, 2):
        d = base / f"fold_{i}"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"x")
    (base / "seed_13").mkdir(parents=True)
    (base / "seed_13" / "best.pt").write_bytes(b"x")

    found = discover_fold_checkpoints(checkpoint_dir)
    names = [p.parent.name for p in found]
    assert names == ["fold_0", "fold_2", "fold_4"], found

    legacy_only = tmp_path / "legacy"
    (legacy_only / "multiclass" / "seed_17").mkdir(parents=True)
    (legacy_only / "multiclass" / "seed_17" / "best.pt").write_bytes(b"x")
    (legacy_only / "multiclass" / "seed_13").mkdir()
    (legacy_only / "multiclass" / "seed_13" / "best.pt").write_bytes(b"x")
    legacy = discover_fold_checkpoints(legacy_only)
    assert [p.parent.name for p in legacy] == ["seed_13"]

    empty = tmp_path / "empty"
    (empty / "multiclass").mkdir(parents=True)
    assert discover_fold_checkpoints(empty) == []
