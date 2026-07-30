"""
evaluation/batch package.
"""

from .batch_executor import BatchExecutor, run_batch_evaluation
from .interfaces import IBatchExecutor

__all__ = [
    "BatchExecutor",
    "IBatchExecutor",
    "run_batch_evaluation",
]
