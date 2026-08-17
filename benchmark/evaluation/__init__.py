"""Evaluation engine — compares memory results against gold truth.

Provides MetricEvaluator interface, EvaluationContext, and concrete evaluators
for recall, false-positive rate, temporal accuracy, contradiction resolution,
follow-up accuracy, module accuracy, and ranking metrics (MRR, NDCG, P@K).
"""

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator
from benchmark.evaluation.context import EvaluationContext
from benchmark.evaluation.contradiction_resolution import ContradictionResolutionEvaluator
from benchmark.evaluation.followup_accuracy import FollowUpAccuracyEvaluator
from benchmark.evaluation.module_accuracy import ModuleAccuracyEvaluator
from benchmark.evaluation.ranking import (
    MRREvaluator,
    NDCGEvaluator,
    PrecisionAtKEvaluator,
    RecallAtKEvaluator,
)

__all__ = [
    "ContradictionResolutionEvaluator",
    "EvaluationContext",
    "EvaluationResult",
    "FollowUpAccuracyEvaluator",
    "MRREvaluator",
    "MetricEvaluator",
    "ModuleAccuracyEvaluator",
    "NDCGEvaluator",
    "PrecisionAtKEvaluator",
    "RecallAtKEvaluator",
]
