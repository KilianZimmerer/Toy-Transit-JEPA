"""Tests for the circular track environment (env.py)."""

import pytest
import torch

from src.env import CircularTrack


class TestCircularTrackConstants:
    def test_state_size(self):
        assert CircularTrack.STATE_SIZE == 24

    def test_condition_size(self):
        assert CircularTrack.CONDITION_SIZE == 1


class TestCircularTrackReset:
    def test_default_reset_places_trains_evenly(self):
        env = CircularTrack(num_trains=3)
        state = env.reset()
        assert state.shape == (CircularTrack.STATE_SIZE,)
        assert int(state.sum().item()) == 3

    def test_reset_with_valid_state(self):
        env = CircularTrack()
        S = CircularTrack.STATE_SIZE
        given = torch.zeros(S)
        given[0] = 1; given[S // 3] = 1; given[2 * S // 3] = 1
        state = env.reset(given)
        assert torch.equal(state, given)

    def test_reset_invalid_size_raises(self):
        env = CircularTrack()
        with pytest.raises(ValueError, match="size"):
            env.reset(torch.zeros(10))

    def test_reset_non_binary_raises(self):
        env = CircularTrack()
        with pytest.raises(ValueError, match="binary"):
            env.reset(torch.tensor([0.5] * CircularTrack.STATE_SIZE))

    def test_reset_clones_input(self):
        env = CircularTrack()
        given = torch.zeros(CircularTrack.STATE_SIZE)
        given[0] = 1
        env.reset(given)
        given[0] = 0
        assert env.state[0] == 1


class TestCircularTrackStep:
    def test_single_train_moves_forward(self):
        env = CircularTrack(num_trains=1)
        S = CircularTrack.STATE_SIZE
        state = torch.zeros(S)
        state[3] = 1
        env.reset(state)
        new_state, _ = env.step()
        assert new_state[4] == 1
        assert new_state[3] == 0

    def test_train_wraps_around(self):
        env = CircularTrack(num_trains=1)
        S = CircularTrack.STATE_SIZE
        state = torch.zeros(S)
        state[S - 1] = 1
        env.reset(state)
        new_state, _ = env.step()
        assert new_state[0] == 1
        assert new_state[S - 1] == 0

    def test_train_blocked_by_occupancy(self):
        """Cell 3 can't move because cell 4 is occupied in the original state."""
        env = CircularTrack(num_trains=2)
        state = torch.zeros(CircularTrack.STATE_SIZE)
        state[3] = 1; state[4] = 1
        env.reset(state)
        new_state, _ = env.step()
        assert new_state[5] == 1
        assert new_state[3] == 1
        assert new_state[4] == 0
        assert int(new_state.sum().item()) == 2

    def test_train_count_preserved(self):
        env = CircularTrack(num_trains=3)
        env.reset()
        for _ in range(20):
            env.step()
        assert int(env.state.sum().item()) == 3

    def test_step_returns_tuple(self):
        env = CircularTrack()
        env.reset()
        result = env.step()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_evenly_spaced_trains_move_in_lockstep(self):
        """3 trains evenly spaced should all advance together."""
        S = CircularTrack.STATE_SIZE
        env = CircularTrack(num_trains=3)
        spacing = S // 3
        state = torch.zeros(S)
        state[0] = 1; state[spacing] = 1; state[2 * spacing] = 1
        env.reset(state)
        env.step()
        expected = torch.zeros(S)
        expected[1] = 1; expected[spacing + 1] = 1; expected[2 * spacing + 1] = 1
        assert torch.equal(env.state, expected)


class TestGenerateBatch:
    def test_output_shapes(self):
        S = CircularTrack.STATE_SIZE
        x_t, a_t, x_next = CircularTrack.generate_batch(16)
        assert x_t.shape == (16, S)
        assert a_t.shape == (16, 1)
        assert x_next.shape == (16, S)

    def test_train_count_preserved_in_batch(self):
        x_t, _, x_next = CircularTrack.generate_batch(32, num_trains=3)
        for i in range(32):
            assert int(x_t[i].sum().item()) == 3
            assert int(x_next[i].sum().item()) == 3

    def test_binary_values(self):
        x_t, a_t, x_next = CircularTrack.generate_batch(16)
        assert torch.all((x_t == 0) | (x_t == 1))
        assert torch.all((x_next == 0) | (x_next == 1))

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            CircularTrack.generate_batch(0)
