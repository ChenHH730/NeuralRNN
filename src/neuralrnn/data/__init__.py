"""Data layer: unified batch format, dataset classes, open-dataset registry and download cache."""
from .base import BaseDataset, StandardScaler, Trials
from .timeseries_dataset import TimeSeriesDataset
from .custom_dataset import CustomDataset
from .cognitive_task_dataset import CognitiveTaskDataset
from .latent_circuit_dataset import LatentCircuitDataset
from .registry import DATASET_REGISTRY, DatasetSpec, load_dataset

from .trial_dataset import TrialTimeseriesDataset

__all__ = [
    "BaseDataset", "StandardScaler", "Trials",
    "TimeSeriesDataset",
    "TrialTimeseriesDataset",
    "CustomDataset",
    "CognitiveTaskDataset",
    "LatentCircuitDataset",
    "DATASET_REGISTRY", "DatasetSpec", "load_dataset",
]


def __getattr__(name):
    # NeurogymDataset depends on the optional package neurogym; load lazily on demand
    if name == "NeurogymDataset":
        from .neurogym_dataset import NeurogymDataset
        return NeurogymDataset
    if name in ("list_neurogym_datasets", "neurogym_version"):
        from . import neurogym_dataset
        return getattr(neurogym_dataset, name)
    raise AttributeError(name)
