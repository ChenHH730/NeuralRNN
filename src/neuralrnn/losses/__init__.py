"""Loss function library -- reusable loss terms and evaluation metrics.

Most training losses are already encapsulated in train/objectives (paradigm-bound).
This module holds pure loss functions and metric utilities for direct use in notebooks
and training loops without depending on the Trainer framework.

Implemented:
- loss_mse: masked per-trial MSE loss (Paradigm A regression tasks)
- accuracy_general: sign-based two-choice decision accuracy (Paradigm A decision tasks)
"""
from .loss_functions import loss_mse, accuracy_general

__all__ = ["loss_mse", "accuracy_general"]
