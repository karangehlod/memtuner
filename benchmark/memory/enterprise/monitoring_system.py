
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Metric:
    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)

class MonitoringSystem:
    """Comprehensive metrics collection and monitoring."""

    def __init__(self):
        self._metrics: dict[str, list] = {}
        self._alerts: list = []
        self._alert_rules: dict[str, Any] = {}

    def record_metric(self, metric_name: str, value: float, tags: dict | None = None) -> None:
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append({'value': value, 'tags': tags or {}})

    def query_metrics(self, metric_name: str) -> list:
        return self._metrics.get(metric_name, [])

    def check_alert_conditions(self) -> list:
        return self._alerts.copy()

    def get_dashboard_data(self) -> dict[str, Any]:
        return {
            'metrics': {k: len(v) for k, v in self._metrics.items()},
            'active_alerts': len(self._alerts),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            'total_metrics': sum(len(v) for v in self._metrics.values()),
            'unique_metrics': len(self._metrics),
            'total_alerts': len(self._alerts),
        }
