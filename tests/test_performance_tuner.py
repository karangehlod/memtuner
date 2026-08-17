
import pytest
from benchmark.memory.production.performance_tuner import PerformanceTuner

@pytest.fixture
def tuner():
    return PerformanceTuner(target_latency_ms=100.0)

class TestPerformanceTuner:
    def test_init(self, tuner):
        assert tuner.target_latency_ms == 100.0
    
    def test_profile_system(self, tuner):
        profile = tuner.profile_system()
        assert profile.avg_latency_ms > 0
        assert profile.p99_latency_ms > 0
    
    def test_auto_tune(self, tuner):
        result = tuner.auto_tune()
        assert "tuned" in result
    
    def test_apply_optimizations(self, tuner):
        profile = tuner.profile_system()
        tuner.apply_optimizations(profile)
    
    def test_verify_sla(self, tuner):
        sla = tuner.verify_sla()
        assert sla["sla_compliant"]
    
    def test_latency_within_target(self, tuner):
        profile = tuner.profile_system()
        assert profile.p99_latency_ms <= 100.0
    
    def test_throughput_positive(self, tuner):
        profile = tuner.profile_system()
        assert profile.throughput > 0
    
    def test_memory_usage_reasonable(self, tuner):
        profile = tuner.profile_system()
        assert profile.memory_usage_mb > 0
    
    def test_auto_tune_result_valid(self, tuner):
        result = tuner.auto_tune()
        assert "optimizations_applied" in result
    
    def test_profile_consistency(self, tuner):
        p1 = tuner.profile_system()
        p2 = tuner.profile_system()
        assert p1.avg_latency_ms == p2.avg_latency_ms
    
    def test_sla_keys(self, tuner):
        sla = tuner.verify_sla()
        assert "sla_compliant" in sla
        assert "uptime_percent" in sla
        assert "latency_p99_ms" in sla
    
    def test_multiple_profiles(self, tuner):
        for _ in range(5):
            profile = tuner.profile_system()
            assert profile is not None
    
    def test_optimization_count(self, tuner):
        result = tuner.auto_tune()
        assert result["optimizations_applied"] >= 0
    
    def test_custom_target_latency(self):
        tuner = PerformanceTuner(target_latency_ms=50.0)
        assert tuner.target_latency_ms == 50.0
    
    def test_profile_measurements(self, tuner):
        profile = tuner.profile_system()
        assert profile.p99_latency_ms >= profile.avg_latency_ms
    
    def test_uptime_percent_valid(self, tuner):
        sla = tuner.verify_sla()
        assert 0 <= sla["uptime_percent"] <= 100
