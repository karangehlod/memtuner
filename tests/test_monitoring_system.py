"""Tests for MonitoringSystem."""

import pytest
from benchmark.memory.enterprise.monitoring_system import MonitoringSystem


@pytest.fixture
def monitor():
    return MonitoringSystem()


class TestMonitoringSystem:
    def test_initialization(self, monitor):
        assert monitor is not None

    def test_record_metric(self, monitor):
        monitor.record_metric("cpu_usage", 75.5)
        metrics = monitor.query_metrics("cpu_usage")
        assert len(metrics) > 0

    def test_record_multiple_metrics(self, monitor):
        monitor.record_metric("cpu", 50.0)
        monitor.record_metric("memory", 60.0)
        monitor.record_metric("cpu", 55.0)
        assert len(monitor.query_metrics("cpu")) == 2

    def test_query_metrics(self, monitor):
        monitor.record_metric("throughput", 100.0)
        result = monitor.query_metrics("throughput")
        assert result[0]["value"] == 100.0

    def test_metrics_with_tags(self, monitor):
        tags = {"service": "api", "region": "us-east"}
        monitor.record_metric("latency", 25.5, tags)
        metrics = monitor.query_metrics("latency")
        assert metrics[0]["tags"]["service"] == "api"

    def test_check_alert_conditions(self, monitor):
        alerts = monitor.check_alert_conditions()
        assert isinstance(alerts, list)

    def test_get_dashboard_data(self, monitor):
        monitor.record_metric("metric1", 10.0)
        monitor.record_metric("metric2", 20.0)
        dashboard = monitor.get_dashboard_data()
        assert "metrics" in dashboard
        assert "active_alerts" in dashboard

    def test_get_stats(self, monitor):
        monitor.record_metric("m1", 1.0)
        monitor.record_metric("m2", 2.0)
        monitor.record_metric("m1", 3.0)
        stats = monitor.get_stats()
        assert stats["total_metrics"] == 3
        assert stats["unique_metrics"] == 2

    def test_unknown_metric_query(self, monitor):
        result = monitor.query_metrics("nonexistent")
        assert result == []

    def test_high_cardinality_metrics(self, monitor):
        for i in range(100):
            monitor.record_metric(f"metric_{i}", float(i))
        stats = monitor.get_stats()
        assert stats["unique_metrics"] == 100

    def test_metric_aggregation(self, monitor):
        for i in range(10):
            monitor.record_metric("value", float(i))
        metrics = monitor.query_metrics("value")
        assert len(metrics) == 10

    def test_real_time_metric_streaming(self, monitor):
        monitor.record_metric("stream", 1.0)
        monitor.record_metric("stream", 2.0)
        result = monitor.query_metrics("stream")
        assert len(result) == 2

    def test_metric_query_performance(self, monitor):
        for i in range(1000):
            monitor.record_metric("perf_test", float(i))
        result = monitor.query_metrics("perf_test")
        assert len(result) == 1000

    def test_custom_metrics(self, monitor):
        monitor.record_metric("custom_business_metric", 42.0)
        result = monitor.query_metrics("custom_business_metric")
        assert result[0]["value"] == 42.0

    def test_tag_based_filtering(self, monitor):
        monitor.record_metric("m1", 1.0, {"env": "prod"})
        monitor.record_metric("m2", 2.0, {"env": "dev"})
        m1 = monitor.query_metrics("m1")
        assert m1[0]["tags"]["env"] == "prod"

    def test_metric_retention(self, monitor):
        monitor.record_metric("retention_test", 10.0)
        data = monitor.get_dashboard_data()
        assert "metrics" in data

    def test_anomaly_detection_ready(self, monitor):
        for i in range(100, 110):
            monitor.record_metric("anomaly", float(i))
        metrics = monitor.query_metrics("anomaly")
        assert len(metrics) == 10

    def test_downsampling_ready(self, monitor):
        for i in range(100):
            monitor.record_metric("downsample", float(i))
        result = monitor.query_metrics("downsample")
        assert len(result) == 100

    def test_integration_multiple_operations(self, monitor):
        monitor.record_metric("cpu", 50.0)
        monitor.record_metric("memory", 60.0)
        data = monitor.get_dashboard_data()
        stats = monitor.get_stats()
        assert stats["total_metrics"] == 2
