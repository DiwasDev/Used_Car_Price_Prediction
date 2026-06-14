"""
base.py
-------
Abstract base class for all pipeline steps.

Design Pattern: Template Method
  - Every transformer defines fit() and transform() separately.
  - fit_transform() is final — subclasses never override it.
  - This mirrors scikit-learn's Transformer API so steps slot into
    a Pipeline or ColumnTransformer if needed later.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class BaseTransformer(ABC):
    """
    Contract every preprocessing step must satisfy.

    Subclass and implement:
        fit(df)        → learn stats/mappings from training data
        transform(df)  → apply learned transformation (pure, no side-effects on input)
    """

    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """Learn parameters from df. Returns self for chaining."""
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation. Must not mutate the input df."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df).transform(df)
