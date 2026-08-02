"""
optimization/exceptions.py
==========================

Exception hierarchy for the Thumbnail Quality Optimization Layer.
"""


class OptimizationBaseError(Exception):
    """Base exception for all optimization layer errors."""
    pass


class BaselineScoringError(OptimizationBaseError):
    """Raised when original thumbnail baseline scoring fails."""
    pass


class ComparativeScoringError(OptimizationBaseError):
    """Raised when candidate vs original comparative scoring fails."""
    pass


class OrchestrationError(OptimizationBaseError):
    """Raised during optimization loop execution or winner selection."""
    pass


class FeedbackError(OptimizationBaseError):
    """Raised when recording or querying outcome history fails."""
    pass


class AcceptanceError(OptimizationBaseError):
    """Raised when acceptance gate evaluation encounters an unrecoverable failure."""
    pass
