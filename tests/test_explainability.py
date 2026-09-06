from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

sys.modules["torchaudio"] = None

import torch
from transformers.models.siglip.modeling_siglip import SiglipVisionConfig, SiglipVisionModel

from explainability import MedSigLIPExplainableModel, class_logit_gradient
from models import MLPHead


def test_explainability_pipeline():
    # 1. Test analytical gradient
    head = MLPHead(in_dim=1152, hidden_dim=512, out_dim=5, dropout=0.1)
    head.eval()

    z = torch.randn(2, 1152, requires_grad=True)
    logits, g_z = head.class_gradient_dz(z, num_classes=5)

    for b in range(2):
        for k in range(5):
            z_sample = z[b : b + 1]
            head.zero_grad()
            out = head(z_sample)
            g_auto = torch.autograd.grad(out[0, k], z_sample, retain_graph=True)[0]
            assert (g_z[b, k] - g_auto[0]).abs().max().item() < 1e-6

    # 2. Test Vision Model integration
    config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=4,
        num_attention_heads=16,
        image_size=448,
        patch_size=16,
    )
    raw_vision = SiglipVisionModel(config)
    for p in raw_vision.parameters():
        p.requires_grad = False
    raw_vision.eval()

    forward_count = 0
    orig_forward = raw_vision.encoder.forward

    def counting_forward(*args, **kwargs):
        nonlocal forward_count
        forward_count += 1
        return orig_forward(*args, **kwargs)

    raw_vision.encoder.forward = counting_forward

    batch_images = torch.randn(2, 3, 448, 448)

    with torch.no_grad():
        base_emb = raw_vision(batch_images).pooler_output
        base_logits = head(base_emb)
        base_grades = base_logits.argmax(dim=-1)

    with MedSigLIPExplainableModel(encoder=raw_vision, classifier=head) as model:
        model.eval()

        for _, param in model.vision_encoder.named_parameters():
            assert not param.requires_grad

        forward_count = 0
        exp_logits, exp_output = model.forward_with_explainability(batch_images, target_size=(448, 448))

        assert forward_count == 1
        assert (base_logits - exp_logits).abs().max().item() < 1e-6
        assert torch.equal(base_grades, exp_output.predicted_grade)
        assert exp_output.attributions.shape == (2, 5, 448, 448)
        assert exp_output.spatial_activation.shape == (2, 28, 28, 1152)
        assert 0.0 <= exp_output.attributions.min().item() <= 1.0
        assert 0.0 <= exp_output.attributions.max().item() <= 1.0

    print("All explainability tests passed successfully.")


if __name__ == "__main__":
    test_explainability_pipeline()
