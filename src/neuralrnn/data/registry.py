"""Open-dataset registry + unified loading entry point load_dataset().

For each paper included, add a DatasetSpec to DATASET_REGISTRY (copy the URL directly from that paper's
notebook wget / Dataverse link) and specify a loader ("module:function").

Design see ARCHITECTURE §3.2. Download and cache logic is in data/download.py (implemented on demand).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetSpec:
    kind: str                                   # neurogym / timeseries / behavioral / trained_rnn / trajectory
    loader: str | None = None                   # "module:function", returns a dataset object
    url: str | None = None                      # download URL (bare URL / Dataverse / Zenodo / OSF)
    files: dict[str, str] | None = None         # logical_name -> filename
    filename: str | None = None                 # single-file download name
    unpack: str | None = None                   # None / "zip" / "tar"
    sha256: str | None = None                   # checksum
    task: str | None = None                     # neurogym task name
    extra: dict[str, Any] = field(default_factory=dict)


# ------- Example entries (URLs come from each notebook) -------
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    # CNS2023: Lorenz63 reconstruction benchmark (includes train/test/pretrained model)
    "lorenz63": DatasetSpec(
        kind="timeseries",
        url="https://github.com/DurstewitzLab/CNS-2023/raw/main/lorenz-datasets.zip",
        unpack="zip",
        files={"train": "lorenz63_train.npy", "test": "lorenz63_test.npy"},
        loader="neuralrnn.data.timeseries_dataset:TimeSeriesDataset.from_npy",
        extra={"dt": 0.01},
    ),
    # nn-brain: neurogym task (no download needed)
    "perceptual_decision_making": DatasetSpec(
        kind="neurogym", task="PerceptualDecisionMaking-v0",
        loader="neuralrnn.data.neurogym_dataset:NeurogymDataset.from_task",
    ),
    # nn-brain: ParametricWorkingMemory task (DelayComparison, no download needed)
    "delay_comparison": DatasetSpec(
        kind="neurogym", task="DelayComparison-v0",
        loader="neuralrnn.data.neurogym_dataset:NeurogymDataset.from_task",
    ),
    # Low-rank RNN (Harvard Dataverse; enable loader when porting recipe 3)
    "dms_lowrank_rank2": DatasetSpec(
        kind="trained_rnn",
        url="https://dataverse.harvard.edu/api/access/datafile/6963161",
        filename="dms_rank2_500.pt",
        # loader="neuralrnn.models.lowrank.modeling_lowrank:load_network",
    ),
    # Langdon & Engel (2025): cognitive tasks (procedurally generated, no download needed)
    "mante": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "mante"},
    ),
    "siegel_miller": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "mante"},  # alias
    ),
    "dms_continuous": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "dms_continuous"},
    ),
    "wm_angle": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "wm_angle"},
    ),
    "parametric_wm": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "wm_angle"},  # alias
    ),
    "wm_frequency": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "wm_frequency"},
    ),
    "romo": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "wm_frequency"},  # alias
    ),
    # Low-rank RNN tasks (Dubreuil et al. 2022 / Valente et al. 2022)
    "rdm": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "rdm"},
    ),
    "two_afc": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "rdm"},  # alias
    ),
    "raposo": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "raposo"},
    ),
    "dms": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "dms"},
    ),
    "mante2": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "mante2"},
    ),
    "lr_mante": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "lr_mante"},  # deprecated alias of mante2 (warns)
    ),
    "go_nogo": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "go_nogo"},
    ),
    # Kleinman et al. 2025: checkerboard decision task (procedurally generated)
    "checkerboard": DatasetSpec(
        kind="cognitive_task",
        loader="neuralrnn.data.cognitive_task_dataset:CognitiveTaskDataset.from_task",
        extra={"task_name": "checkerboard"},
    ),
    # Tiny RNN: Bartolo Monkey probabilistic reversal-learning task behavioral data
    "bartolo_monkey": DatasetSpec(
        kind="behavioral",
        loader="neuralrnn.data.bartolo_monkey_dataset:BartoloMonkeyDataset.load",
        extra={"animal_name": "V"},
    ),
    # Append new paper entries here ...
}


def _resolve(spec_loader: str):
    """Resolve 'module:attr' or 'module:Class.method' to a callable."""
    module_path, attr = spec_loader.split(":")
    obj = importlib.import_module(module_path)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_dataset(name: str, **overrides):
    """Unified entry point: lookup registry -> (download/cache on demand) -> instantiate dataset.

    overrides are forwarded to loader (e.g., sequence_length / batch_size / dt / seq_len).

    Names not found in DATASET_REGISTRY fall through to neurogym: any env id registered by the
    installed neurogym (e.g. 'GoNogo-v0'; case-insensitive, '-v0' optional) loads as a
    NeurogymDataset. See neuralrnn.data.list_neurogym_datasets() for what is available.
    """
    if name not in DATASET_REGISTRY:
        # Dynamic passthrough to neurogym env ids (registered names always win, so built-in
        # tasks like 'go_nogo' are never shadowed; use the env id 'GoNogo-v0' for neurogym's).
        from .neurogym_dataset import NeurogymDataset, _resolve_task_id, list_neurogym_datasets
        resolved = _resolve_task_id(name)
        if resolved in list_neurogym_datasets():
            return NeurogymDataset.from_task(task=resolved, **overrides)
        raise KeyError(
            f"Dataset '{name}' is not registered. Available: {sorted(DATASET_REGISTRY)}. "
            f"Any env id from the installed neurogym (e.g. 'GoNogo-v0') also works; "
            f"see neuralrnn.data.list_neurogym_datasets()."
        )
    spec = DATASET_REGISTRY[name]

    if spec.kind == "neurogym":
        loader = _resolve(spec.loader)
        return loader(task=spec.task, **overrides)

    if spec.kind == "cognitive_task":
        loader = _resolve(spec.loader)
        task_name = spec.extra.get("task_name", name)
        return loader(task_name=task_name, **overrides)

    if spec.kind == "behavioral":
        loader = _resolve(spec.loader)
        extra = {**spec.extra, **overrides}
        return loader(**extra)

    # Local file datasets that require download
    from .download import ensure_files  # Implemented on demand: returns {logical_name: local path}
    local = ensure_files(spec)          # Handles download / unpack / verification / cache
    loader = _resolve(spec.loader)
    if spec.files:
        # Convention: logical names in files (train/test...) are passed as *_path keywords
        path_kwargs = {f"{k}_path": v for k, v in local.items()}
        extra = {**(spec.extra or {}), **overrides}
        return loader(**path_kwargs, **extra)
    return loader(local.get("file"), **overrides)
