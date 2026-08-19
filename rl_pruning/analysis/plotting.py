import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history, output_dir="training_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Log Data to CSV
    csv_path = os.path.join(output_dir, "training_log.csv")
    df = pd.DataFrame(
        {
            "episode": range(len(history["rewards"])),
            "reward": history["rewards"],
            "accuracy": history["accuracies"],
            "compression": history["compressions"],
        }
    )

    # Full-eval accuracy (sparse, measured every eval_every episodes)
    full_acc = history.get("full_accuracies", [])
    if full_acc:
        full_ep, full_vals = zip(*full_acc)
        df["full_accuracy"] = np.nan
        df.loc[list(full_ep), "full_accuracy"] = full_vals

    alpha = history.get("alpha", 0.0)
    df["alpha"] = alpha
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV log saved to {csv_path}")

    # Generate Plots
    episodes = df["episode"].values
    window = max(1, len(episodes) // 10)

    metrics = [
        ("reward", "Episode Reward", "Reward", "reward.png", "blue"),
        ("accuracy", "Final Accuracy", "Accuracy Ratio", "accuracy.png", "green"),
        (
            "compression",
            "Compression Ratio",
            "Params / Original",
            "compression.png",
            "purple",
        ),
    ]

    for key, title, ylabel, filename, color in metrics:
        plt.figure(figsize=(8, 5))
        data = df[key].values

        plt.plot(episodes, data, alpha=0.3, color=color, label="Raw")

        if len(data) > window:
            ma_data = np.convolve(data, np.ones(window) / window, mode="valid")
            plt.plot(
                range(window - 1, len(episodes)),
                ma_data,
                color="red",
                linewidth=2,
                label="Trend",
            )

        if key == "accuracy":
            plt.axhline(y=data[0], color="black", linestyle="--", label="Baseline")
            if full_acc:
                full_ep, full_vals = zip(*full_acc)
                plt.plot(
                    list(full_ep),
                    list(full_vals),
                    color="orange",
                    marker="o",
                    linestyle="",
                    label="Full eval",
                )

        plt.title(f"{title} (α={alpha:.3f})")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.2)

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Plot saved: {save_path}")
