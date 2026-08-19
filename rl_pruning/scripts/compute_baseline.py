import json
import torch
from pathlib import Path
from copy import deepcopy

from rl_pruning.pruning.uniform import uniform_prune
from rl_pruning.eval.glue_evaluator import GLUEEvaluator
from rl_pruning.utils.model_loading import load_sst2_model

# TODO: add more if needed for interpolation
COMPRESSIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def main():
    device = "cuda" if torch.cuda.is_available() else "mps"

    model, tokenizer = load_sst2_model()
    model.to(device)

    evaluator = GLUEEvaluator(
        model=model,
        tokenizer=tokenizer,
        task_name="sst2",
        device=device,
        batch_size=32,
    )

    baseline = {}

    for c in COMPRESSIONS:
        m = deepcopy(model)

        if c > 0:
            uniform_prune(m, target_compression=c)

        m.to(device)
        acc = evaluator(m)
        baseline[c] = acc

        print(f"Compression {c:.2f} → acc {acc:.4f}")

    out_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "baselines"
        / "bert_base_sst2_uniform.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Saved baseline to {out_path}")


if __name__ == "__main__":
    main()

