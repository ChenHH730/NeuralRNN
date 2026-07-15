"""Tests for reusable loss functions, regularizers, and metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralrnn.modeling_utils import DynamicsModelOutput
from neuralrnn.train.losses import (
    masked_mse,
    masked_cross_entropy,
    masked_nll,
    loss_mse,
    activity_l2,
    weight_l2,
    weight_l1,
    orthogonality_penalty,
    model_orthogonality_penalty,
    accuracy_classification,
    accuracy_general,
)


class TestMaskedMSE:
    def test_no_mask(self):
        y = torch.ones(2, 3, 4)
        t = torch.zeros(2, 3, 4)
        loss = masked_mse(y, t, None)
        assert loss.shape == ()
        assert loss.item() == pytest.approx(1.0)

    def test_with_mask(self):
        y = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1,2,2)
        t = torch.zeros_like(y)
        mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
        loss = masked_mse(y, t, mask)
        # counted positions: 1, 0, 9, 16 -> sum=26, count=4 -> mean=6.5; per-trial: (1+0)/2 + (9+16)/2 = 0.5+12.5 = 13; /1 trial = 13
        expected = (1.0 + 9.0 + 16.0) / 3.0
        assert loss.item() == pytest.approx(expected)

    def test_dynamics_model_output(self):
        y = DynamicsModelOutput(outputs=torch.ones(2, 3, 4))
        t = torch.zeros(2, 3, 4)
        loss = masked_mse(y, t, None)
        assert loss.item() == pytest.approx(1.0)

    def test_loss_mse_alias(self):
        y = torch.ones(2, 3, 4)
        t = torch.zeros(2, 3, 4)
        assert loss_mse(y, t, None).item() == masked_mse(y, t, None).item()

    def test_global_reduction_matches_notebook_convention(self):
        y = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        t = torch.zeros_like(y)
        mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 1.0]]])
        loss = masked_mse(y, t, mask, reduction="global")
        expected = ((y ** 2) * mask).sum() / mask.sum()
        assert loss.item() == pytest.approx(expected.item())


class TestMaskedCrossEntropy:
    def test_no_mask(self):
        logits = torch.zeros(2, 3, 5)
        targets = torch.zeros(2, 3, dtype=torch.long)
        loss = masked_cross_entropy(logits, targets, None)
        # uniform logits -> CE = log(5)
        assert loss.item() == pytest.approx(np.log(5.0), abs=1e-5)

    def test_with_mask(self):
        logits = torch.zeros(2, 3, 5)
        targets = torch.zeros(2, 3, dtype=torch.long)
        mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        loss = masked_cross_entropy(logits, targets, mask)
        # 4 valid positions out of 6
        assert loss.item() == pytest.approx(np.log(5.0), abs=1e-5)

    def test_nll_alias(self):
        logits = torch.zeros(2, 3, 5)
        targets = torch.zeros(2, 3, dtype=torch.long)
        assert masked_nll(logits, targets, None).item() == masked_cross_entropy(logits, targets, None).item()


class TestActivityL2:
    def test_basic(self):
        states = torch.ones(2, 3, 4)
        assert activity_l2(states).item() == pytest.approx(1.0)

    def test_masked(self):
        states = torch.ones(1, 3, 2)
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        assert activity_l2(states, mask).item() == pytest.approx(1.0)

    def test_global_reduction_ignores_mask(self):
        states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.zeros(1, 2)
        # global reduction ignores the mask and returns mean over all elements
        assert activity_l2(states, mask, reduction="global").item() == pytest.approx(
            (states ** 2).mean().item()
        )


class TestWeightRegularizers:
    def test_l2(self):
        model = torch.nn.Linear(3, 2)
        torch.nn.init.constant_(model.weight, 1.0)
        torch.nn.init.constant_(model.bias, 0.0)
        val = weight_l2(model).item()
        # 6 weights + 2 biases = 8 params, all squared sum = 6; mean = 6/8 = 0.75
        assert val == pytest.approx(0.75)

    def test_l2_with_patterns(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 2, bias=False),
            torch.nn.Linear(2, 1, bias=False),
        )
        torch.nn.init.constant_(model[0].weight, 1.0)
        torch.nn.init.constant_(model[1].weight, 2.0)
        # all params: 6*1 + 2*4 = 14 / 8 = 1.75
        assert weight_l2(model).item() == pytest.approx(1.75)
        # only first layer: 6 / 6 = 1
        assert weight_l2(model, ["0\\.weight"]).item() == pytest.approx(1.0)

    def test_l1(self):
        model = torch.nn.Linear(3, 2)
        torch.nn.init.constant_(model.weight, -1.0)
        torch.nn.init.constant_(model.bias, 2.0)
        # abs sum = 6 + 4 = 10; count = 8 -> 1.25
        assert weight_l1(model).item() == pytest.approx(1.25)

    def test_sum_reduction_matches_notebook_convention(self):
        model = torch.nn.Linear(3, 2)
        torch.nn.init.constant_(model.weight, 1.0)
        torch.nn.init.constant_(model.bias, 0.0)
        # 6 weights squared and summed = 6
        assert weight_l2(model, reduction="sum").item() == pytest.approx(6.0)
        # mean reduction is still the default
        assert weight_l2(model).item() == pytest.approx(6.0 / 8.0)


class TestOrthogonalityPenalty:
    def test_perfectly_orthogonal(self):
        eye = torch.eye(3)
        # input weight = first two columns, output weight = last column transposed
        w_in = eye[:, :2]
        w_out = eye[:, 2:].t()  # (1, 3)
        pen = orthogonality_penalty(w_in, w_out)
        # B = [e1 e2 e3] is orthonormal -> B^T B = I -> off-diagonal zero
        assert pen.item() == pytest.approx(0.0, abs=1e-5)

    def test_model_orthogonality_missing_attributes(self):
        model = torch.nn.Linear(3, 2)
        pen = model_orthogonality_penalty(model, "missing_input", "missing_output")
        assert pen.item() == 0.0


class TestAccuracyClassification:
    def test_perfect(self):
        logits = torch.zeros(2, 3, 5)
        logits[:, :, 0] = 10.0
        targets = torch.zeros(2, 3, dtype=torch.long)
        assert accuracy_classification(logits, targets).item() == pytest.approx(1.0)

    def test_half(self):
        logits = torch.zeros(2, 3, 2)
        logits[:, :, 0] = 1.0
        targets = torch.zeros(2, 3, dtype=torch.long)
        targets[0, 0] = 1  # one wrong
        assert accuracy_classification(logits, targets).item() == pytest.approx(5.0 / 6.0)


class TestAccuracyGeneral:
    def test_sign_accuracy(self):
        output = torch.tensor([[[1.0], [2.0]], [[-1.0], [-2.0]]])
        targets = torch.tensor([[[1.0], [2.0]], [[-1.0], [-2.0]]])
        mask = torch.ones_like(output)
        assert accuracy_general(output, targets, mask).item() == pytest.approx(1.0)

    def test_no_valid_trials(self):
        output = torch.ones(2, 3, 1)
        targets = torch.zeros(2, 3, 1)
        mask = torch.ones_like(output)
        assert torch.isnan(accuracy_general(output, targets, mask))
