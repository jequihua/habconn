"""Unit tests for the v1 reward module."""

from habconn.evaluators.reward import compute_delta_pc_reward, RewardResult


def test_positive_delta():
    r = compute_delta_pc_reward(pc_before=1.0e-5, pc_after=1.5e-5)
    assert isinstance(r, RewardResult)
    assert r.reward > 0
    assert r.delta_pc == r.pc_after - r.pc_before
    assert r.reward == r.delta_pc


def test_zero_delta():
    r = compute_delta_pc_reward(pc_before=2.0e-5, pc_after=2.0e-5)
    assert r.reward == 0.0
    assert r.delta_pc == 0.0


def test_negative_delta():
    r = compute_delta_pc_reward(pc_before=2.0e-5, pc_after=1.0e-5)
    assert r.reward < 0
    assert r.delta_pc < 0
