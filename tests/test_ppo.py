import torch
import pytest

from rl_pruning.models.ppo import PPOActorCritic


@pytest.fixture
def net():
    torch.manual_seed(0)
    return PPOActorCritic(obs_dim=5, act_dim=2, max_action=0.8)


def test_act_and_evaluate_logp_consistent(net):
    """Stored logp from act() must match evaluate() on the same rollout."""
    obs = torch.randn(64, 5)
    action, u, logp_act, _ = net.act(obs)
    logp_eval, _, _ = net.evaluate(obs, u)
    assert torch.allclose(logp_act, logp_eval, atol=1e-4)


def test_actions_bounded_to_zero_max(net):
    obs = torch.randn(256, 5)
    for bias in (-10.0, 10.0, 0.0):
        net.mu.bias.data.fill_(bias)
        action, _, _, _ = net.act(obs)
        assert (action >= 0.0).all(), f"negative action at bias={bias}"
        assert (action <= 0.8).all(), f"action exceeds max at bias={bias}"


def test_logp_finite_no_nan(net):
    obs = torch.randn(64, 5)
    action, u, _, _ = net.act(obs)
    _, logp_eval, _ = net.evaluate(obs, u)
    assert not torch.isnan(logp_eval).any()
    assert not torch.isinf(logp_eval).any()


def test_logp_consistent_even_at_saturation(net):
    """Storing u makes logp exact even when tanh saturates (no inversion)."""
    obs = torch.randn(64, 5)
    for bias in (-10.0, 10.0, 20.0, -20.0, 0.0, 3.0):
        net.mu.bias.data.fill_(bias)
        action, u, logp_act, _ = net.act(obs)
        logp_eval, _, _ = net.evaluate(obs, u)
        assert torch.allclose(logp_act, logp_eval, atol=1e-4), f"bias={bias}"


def test_logp_finite_at_saturation(net):
    """Extreme saturation must not produce inf/NaN (stable Jacobian)."""
    obs = torch.randn(64, 5)
    for bias in (10.0, -10.0, 20.0, -20.0):
        net.mu.bias.data.fill_(bias)
        _, u, logp_act, _ = net.act(obs)
        logp_eval, _, _ = net.evaluate(obs, u)
        assert not torch.isinf(logp_act).any(), f"act() overflowed at bias={bias}"
        assert not torch.isinf(logp_eval).any(), f"evaluate() overflowed at bias={bias}"
        assert not torch.isnan(logp_act).any(), f"act() NaN at bias={bias}"
        assert not torch.isnan(logp_eval).any(), f"evaluate() NaN at bias={bias}"


def test_log_jacobian_correct_constant(net):
    """Jacobian constant term is log(max/2), not log(2/max)."""
    u = torch.zeros(1, 2)
    jac = net._log_jacobian(u)
    # a = 0.5*max*(tanh(0)+1) = 0.5*max; da/du = 0.5*max*sech^2(0) = 0.5*max
    # log|da/du| = log(max/2) = log(0.4) = -0.916 (per dim, summed over 2 dims)
    expected = torch.log(torch.tensor(0.8 / 2.0)) * 2.0
    assert jac.item() == pytest.approx(expected.item())


def test_forward_output_shapes(net):
    obs = torch.randn(10, 5)
    mu, v = net.forward(obs)
    assert mu.shape == (10, 2)
    assert v.shape == (10, 1)
