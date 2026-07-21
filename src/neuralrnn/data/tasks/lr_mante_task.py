"""Deprecated module — the task moved to :mod:`mante2_task` (registry name
``mante2``; ``lr_mante`` remains as a deprecated alias).

This module IS ``mante2_task`` (module alias): every attribute — including the
legacy module-level timing globals (``total_duration``, ``stim_begin``, ...)
and ``_setup()`` — resolves there, so mutations such as
``lr_mante_task.SCALE_CTX = 0.5`` propagate to trial generation exactly as
before the refactor.
"""
import sys as _sys

from . import mante2_task as _mante2_task

_sys.modules[__name__] = _mante2_task
