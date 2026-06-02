"""Latent circuit dataset — pre-computed RNN data for latent circuit fitting.

Holds task inputs, RNN behavioral outputs, and RNN hidden states.
Used to fit the latent circuit model to trained RNN responses.
"""
from __future__ import annotations

import torch

from .base import BaseDataset


class LatentCircuitDataset(BaseDataset):
    """Dataset for fitting latent circuit models to RNN responses.

    Stores pre-computed data from running a trained RNN on task inputs:
    - inputs: task inputs (from the cognitive task)
    - targets: RNN behavioral output (readout of hidden states)
    - rnn_states: RNN hidden state trajectories

    The latent circuit model is trained to:
    1. Produce the same behavioral output (MSE on targets)
    2. Have embedded latent states match RNN states (NMSE via embedding Q)

    Attributes:
        kind: "latent_circuit"
        inputs: (N, T, K) tensor — task inputs
        targets: (N, T, output_dim) tensor — RNN behavioral output
        rnn_states: (N, T, N_rnn) tensor — RNN hidden states
        batch_size: batch size for sample_batch()
    """

    kind = "latent_circuit"

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        rnn_states: torch.Tensor,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.rnn_states = rnn_states
        self.batch_size = batch_size
        self.input_dim = inputs.shape[-1]
        self.output_dim = targets.shape[-1]

    @classmethod
    def from_rnn_and_task(
        cls,
        rnn_model,
        task_dataset,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> "LatentCircuitDataset":
        """Create dataset by running a trained RNN on task data.

        Args:
            rnn_model: Trained RNN model (NeuralDynamicsModel).
            task_dataset: CognitiveTaskDataset with task inputs.
            batch_size: Batch size for sample_batch().
            device: Device to run the RNN on.

        Returns:
            LatentCircuitDataset instance.
        """
        rnn_model.eval()
        with torch.no_grad():
            data = task_dataset.get_all_trials()
            inputs = data["inputs"].to(device)
            # Run RNN forward to get hidden states
            out = rnn_model(inputs)
            rnn_states = out.states  # (N, T, latent_dim)
            # Behavioral output: readout of hidden states
            targets = out.outputs  # (N, T, output_dim)

        return cls(
            inputs=inputs.cpu(),
            targets=targets.cpu(),
            rnn_states=rnn_states.cpu(),
            batch_size=batch_size,
        )

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a random batch.

        Returns:
            dict with keys:
                "inputs":     (B, T, K) — task inputs
                "targets":    (B, T, output_dim) — RNN behavioral output
                "rnn_states": (B, T, N) — RNN hidden states
        """
        n = self.inputs.shape[0]
        idx = torch.randint(0, n, (self.batch_size,))
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx],
            "rnn_states": self.rnn_states[idx],
        }

    def __len__(self) -> int:
        return self.inputs.shape[0]
