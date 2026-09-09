"""Auditable experiment preparation and resumable stage transactions.

These primitives do not imply a fully connected SpatialCraft learning runtime.
"""

from .journal import RunJournal
from .protocol import split_category, stratified_halves

__all__ = ["RunJournal", "split_category", "stratified_halves"]
