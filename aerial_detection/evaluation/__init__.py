"""Evaluation module for aerial object detection."""

from .metrics import (
    DOTAEvaluator,
    EvaluationResult,
    GroundTruth,
    Prediction,
    compute_map
)

__all__ = [
    'DOTAEvaluator',
    'EvaluationResult',
    'GroundTruth',
    'Prediction',
    'compute_map'
]
