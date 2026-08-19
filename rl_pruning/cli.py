import typer
import torch

from rl_pruning.config import PPOConfig, PruningConfig
from rl_pruning.envs.transformer_pruning_env import TransformerPruningEnv
from rl_pruning.eval.glue_evaluator import GLUEEvaluator
from rl_pruning.eval.partial_evaluator import PartialEvaluator
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
    alpha: float = 0.3,
    max_prune_ratio: float = 0.8,
    subset_frac: float = 0.1,
    eval_every: int = 10,
    n_rollouts: int = 8,
    acc_floor_ratio: float = 0.85,
    shaping_coef: float = 1.0,
    shaping_penalty: float = 0.1,
    layer_weight_decay: float = 0.0,
):
    device = get_device()
    typer.echo(f"Using device: {device}")

    model, tokenizer = load_sst2_model()
    evaluator = GLUEEvaluator(model, tokenizer, device=device)

    pruning = PruningConfig(
        max_prune_ratio=max_prune_ratio,
        alpha=alpha,
        acc_floor_ratio=acc_floor_ratio,
        shaping_coef=shaping_coef,
        shaping_penalty=shaping_penalty,
    )

    env = TransformerPruningEnv(
        model=model,
        tokenizer=tokenizer,
        eval_fn=PartialEvaluator(evaluator, subset_frac=subset_frac),
        max_prune_ratio=pruning.max_prune_ratio,
        alpha=pruning.alpha,
        acc_floor_ratio=pruning.acc_floor_ratio,
        shaping_coef=pruning.shaping_coef,
        shaping_penalty=pruning.shaping_penalty,
        device=device,
    )

    config = PPOConfig(n_rollouts=n_rollouts, layer_weight_decay=layer_weight_decay)

    agent, history = train_ppo(
        env=env,
        config=config,
        episodes=episodes,
        device=device,
        full_eval_fn=evaluator,
        eval_every=eval_every,
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
