import typer
import torch

from rl_pruning.config import PPOConfig, PruningConfig
from rl_pruning.envs.transformer_pruning_env import TransformerPruningEnv
from rl_pruning.eval.glue_evaluator import GLUEEvaluator
from rl_pruning.training.trainer import train_ppo
from rl_pruning.analysis.policy_analysis import analyze_pruning_policy
from rl_pruning.analysis.plotting import plot_training_curves
from rl_pruning.utils.model_loading import load_sst2_model

app = typer.Typer(help="RL-based Transformer pruning")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def train(
    episodes: int = 200,
    alpha: float = 0.05,
    max_prune_ratio: float = 0.8,
):
    device = get_device()
    typer.echo(f"Using device: {device}")

    model, tokenizer = load_sst2_model()
    evaluator = GLUEEvaluator(model, tokenizer, device=device)

    env = TransformerPruningEnv(
        model=model,
        tokenizer=tokenizer,
        eval_fn=evaluator,
        max_prune_ratio=max_prune_ratio,
        alpha=alpha,
        device=device,
    )

    agent, history = train_ppo(
        env=env,
        config=PPOConfig(),
        episodes=episodes,
        device=device,
    )

    plot_training_curves(history)
    analyze_pruning_policy(agent, env, device=device)


@app.command()
def validate(
    episodes: int = 5,
):
    device = get_device()
    model, tokenizer = load_sst2_model()
    evaluator = GLUEEvaluator(model, tokenizer, device=device)

    env = TransformerPruningEnv(
        model=model,
        tokenizer=tokenizer,
        eval_fn=evaluator,
        device=device,
    )

    from rl_pruning.envs.transformer_pruning_env import validate_environment

    validate_environment(env, n_test_episodes=episodes)


if __name__ == "__main__":
    app()
