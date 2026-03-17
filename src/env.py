"""Circular track environment with deterministic train dynamics."""

import torch


class CircularTrack:
    """Circular 1D track where trains move forward, wrapping around."""

    STATE_SIZE: int = 24
    CONDITION_SIZE: int = 1  # dummy condition (unused, keeps JEPA interface)

    def __init__(self, num_trains: int = 3) -> None:
        self.num_trains = num_trains
        self.state = torch.zeros(self.STATE_SIZE)

    def reset(self, state: torch.Tensor | None = None) -> torch.Tensor:
        """Reset to given state or place trains evenly spaced."""
        if state is not None:
            if state.shape != (self.STATE_SIZE,):
                raise ValueError(
                    f"State must be a tensor of size {self.STATE_SIZE}, "
                    f"got shape {state.shape}"
                )
            if not torch.all((state == 0) | (state == 1)):
                raise ValueError("State must contain only binary values (0 or 1)")
            self.state = state.clone().float()
        else:
            # Place trains evenly spaced around the circle
            self.state = torch.zeros(self.STATE_SIZE)
            spacing = self.STATE_SIZE // self.num_trains
            for i in range(self.num_trains):
                self.state[i * spacing] = 1
        return self.state

    def step(self, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one timestep. Each train moves forward by 1 if next cell is empty.

        Movement wraps around: cell (STATE_SIZE-1) -> cell 0.
        Process backward to avoid double-moves.
        """
        if condition is None:
            condition = torch.zeros(self.CONDITION_SIZE)

        new_state = torch.zeros_like(self.state)

        # Process each cell: if occupied, try to move forward
        # We need to handle wrapping, so collect moves first
        moved = [False] * self.STATE_SIZE
        for i in range(self.STATE_SIZE - 1, -1, -1):
            if self.state[i] == 1:
                next_cell = (i + 1) % self.STATE_SIZE
                if self.state[next_cell] == 0 and new_state[next_cell] == 0:
                    new_state[next_cell] = 1
                    moved[i] = True
                else:
                    new_state[i] = 1

        self.state = new_state
        return self.state, condition

    @staticmethod
    def generate_batch(
        batch_size: int, num_trains: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate batch of (x_t, a_t, x_{t+1}) transitions.

        Each uses a random valid state with exactly num_trains trains.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        state_size = CircularTrack.STATE_SIZE
        cond_size = CircularTrack.CONDITION_SIZE

        x_t_list = []
        a_t_list = []
        x_next_list = []

        for _ in range(batch_size):
            env = CircularTrack(num_trains=num_trains)
            # Random valid state: pick num_trains random positions
            state = torch.zeros(state_size)
            positions = torch.randperm(state_size)[:num_trains]
            state[positions] = 1
            env.reset(state)

            condition = torch.zeros(cond_size)
            x_t_list.append(env.state.clone())
            next_state, _ = env.step(condition)
            a_t_list.append(condition)
            x_next_list.append(next_state.clone())

        return torch.stack(x_t_list), torch.stack(a_t_list), torch.stack(x_next_list)
