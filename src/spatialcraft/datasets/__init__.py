"""Dataset normalization and adapter API."""

from .base import (
    DatasetAdapter,
    DatasetConfigurationError,
    DatasetError,
    DatasetFormatError,
    DatasetInfo,
    RecordDatasetAdapter,
    UnknownSplitError,
)
from .normalizer import ImageMaterializer
from .registry import DatasetRegistry, create_default_registry
from .split_manager import SplitManager, task_category

__all__ = [
    "DatasetAdapter",
    "DatasetConfigurationError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetInfo",
    "DatasetRegistry",
    "ImageMaterializer",
    "RecordDatasetAdapter",
    "SplitManager",
    "UnknownSplitError",
    "create_default_registry",
    "task_category",
]
