import json
import numpy as np
from pathlib import Path


def load_baseline_curve():
    path = (
        Path(__file__).parent.parent
        / "assets"
        / "baselines"
        / "tinybert_base_sst2_uniform.json"
    )

    with open(path) as f:
        data = json.load(f)

    return {float(k): float(v) for k, v in data.items()}


# TODO: naive based on layer weight density
def compute_layer_importance(model):
    importance = []
    for layer in model.bert.encoder.layer:
        w = layer.output.dense.weight
        importance.append(w.abs().mean().item())

    importance = np.array(importance, dtype=np.float32)
    importance /= importance.mean()  # normalize
    return importance.tolist()
