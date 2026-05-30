"""数据层：统一 batch 格式、数据集类、开源数据集注册表与下载缓存。"""
from .base import BaseDataset, StandardScaler
from .timeseries_dataset import TimeSeriesDataset
from .custom_dataset import CustomDataset
from .registry import DATASET_REGISTRY, DatasetSpec, load_dataset

__all__ = [
    "BaseDataset", "StandardScaler",
    "TimeSeriesDataset",
    "CustomDataset",
    "DATASET_REGISTRY", "DatasetSpec", "load_dataset",
]


def __getattr__(name):
    # NeurogymDataset 依赖可选包 neurogym，按需懒加载
    if name == "NeurogymDataset":
        from .neurogym_dataset import NeurogymDataset
        return NeurogymDataset
    raise AttributeError(name)
