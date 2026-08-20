import numpy as np


# TODO: naive based on layer weight density
def compute_layer_importance(model):
    importance = []
    for layer in model.bert.encoder.layer:
        w = layer.output.dense.weight
        importance.append(w.abs().mean().item())

    importance = np.array(importance, dtype=np.float32)
    importance /= importance.mean()  # normalize
    return importance.tolist()
