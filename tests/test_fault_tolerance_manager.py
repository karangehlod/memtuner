"""Comprehensive tests for FaultToleranceManager."""

import pytest
import time
from unittest.mock import Mock
from benchmark.memory.distributed.fault_tolerance_manager import (
    FaultToleranceManager,
    RetryStrategy,
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


@pytest.fixture
def manager():
    """Create fault tolerance manager."""
    return FaultToleranceManager()


@pytest.fixture
def retry_config():
    """Create retry config."""
    return RetryConfig(
        max_attempts=3,
        initial_delay_sec=0.01,
        max_delay_sec=0.1,
        backoff_multiplier=2.0,
    )


@pytest.fixture
def circuit_breaker_config():
    """Create circuit breaker config."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_sec=1.0,
        success_threshold=2,
    )


class TestRetryStrategy:
    """Test retry strategy."""

    def test_retry_strategy_initialization(self, retry_config):
        """Test retry strategy initialization."""
        strategy = RetryStrategy(retry_config)

        assert strategy.attempt == 0
        assert not strategy.is_exhausted()

    def test_should_retry_timeout(self, retry_config):
        """Test retry on timeout."""
        strategy = RetryStrategy(retry_config)

        exc = TimeoutError("Connection timeout")
        assert strategy.should_retry(exc)

    def test_should_retry_connection(self, retry_config):
        """Test retry on connection error."""
        strategy = RetryStrategy(retry_config)

        exc = ConnectionError("Connection failed")
        assert strategy.should_retry(exc)

    def test_should_not_retry_value_error(self, retry_config):
        """Test no retry on value error."""
        strategy = RetryStrategy(retry_config)

        exc = ValueError("Invalid value")
        assert not strategy.should_retry(exc)

    def test_backoff_delay_increases(self, retry_config):
        """Test backoff delay increases exponentially."""
        strategy = RetryStrategy(retry_config)

        delay1 = strategy.get_backoff_delay()
        delay2 = strategy.get_backoff_delay()
        delay3 = strategy.get_backoff_delay()

        # Delays should increase (accounting for jitter)
        assert delay2 > delay1 * 0.8  # Allow for jitter
        assert delay3 > delay2 * 0.8

    def test_max_delay_respected(self, retry_config):
        """Test max delay is respected."""
        strategy = RetryStrategy(retry_config)

        # Get delays multiple times
        for _ in range(10):
            delay = strategy.get_backoff_delay()
            # Allow small tolerance for jitter
            assert delay <= retry_config.max_delay_sec * 1.2

    def test_retry_exhaustion(self, retry_config):
        """Test retry exhaustion detection."""
        strategy = RetryStrategy(retry_config)

        assert not strategy.is_exhausted()

        for i in range(retry_config.max_attempts):
            strategy.get_backoff_delay()

        assert strategy.is_exhausted()

    def test_retry_reset(self, retry_config):
        """Test retry reset."""
        strategy = RetryStrategy(retry_config)

        for _ in range(retry_config.max_attempts):
            strategy.get_backoff_delay()

        assert strategy.is_exhausted()

        strategy.reset()
        assert not strategy.is_exhausted()


class TestCircuitBreaker:
    """Test circuit breaker."""

    def test_circuit_breaker_initialization(self, circuit_breaker_config):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(1, circuit_breaker_config)

        state = breaker.get_state()
        assert state.state == CircuitState.CLOSED
        assert state.failure_count == 0

    def test_circuit_starts_closed(self, circuit_breaker_config):
        """Test circuit starts in CLOSED state."""
        breaker = CircuitBreaker(1, circuit_breaker_config)

        assert breaker.can_execute()
        assert breaker.get_state().state == CircuitState.CLOSED

    def test_circuit_opens_after_failures(self, circuit_breaker_config):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(1, circuit_breaker_config)

        # Record failures
        for _ in range(circuit_breaker_config.failure_threshold):
            breaker.record_failure()

        # Circuit should be OPEN
        assert not breaker.can_execute()
        assert breaker.get_state().state == CircuitState.OPEN

    def test_circuit_half_open_after_timeout(self, circuit_breaker_config):
        """Test circuit goes to HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(1, circuit_breaker_config)

        # Open the circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            breaker.record_failure()

        # Wait for recovery timeout
        time.sleep(circuit_breaker_config.recovery_timeout_sec + 0.1)

        # Should be able to execute (testing)
        assert breaker.can_execute()
        assert breaker.get_state().state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_successes(self, circuit_breaker_config):
        """Test circuit closes after successes in HALF_OPEN."""
        breaker = CircuitBreaker(1, circuit_breaker_config)

        # Open the circuit
        for _ in range(circuit_breaker_config.failure_threshold):
            breaker.record_failure()

        # Force to HALF_OPEN
        time.sleep(circuit_breaker_config.recovery_timeout_sec + 0.1)
        breaker.can_execute()

        # Record successes
        for _ in range(circuit_breaker_config.success_threshold):
            breaker.record_success()

        # Circuit should be CLOSED
        assert breaker.get_state().state == CircuitState.CLOSED

    def test_circuit_reopens_on_failure_in_half_open(self, circuit_breaker_config):
        """Test circuit reopens if fails while testing."""
        breaker = CircuitBreaker(1, circuit_breaker_config)

        # Open -> HALF_OPEN
        for _ in range(circuit_breaker_config.failure_threshold):
            breaker.record_failure()

        time.sleep(circuit_breaker_config.recovery_timeout_sec + 0.1)
        breaker.can_execute()

        # Fail while testing
        breaker.record_failure()

        assert breaker.get_state().state == CircuitState.OPEN


class TestFaultToleranceManagerInitialization:
    """Test fault tolerance manager initialization."""

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert len(manager.get_error_log()) == 0

    def test_initialization_with_configs(self, retry_config, circuit_breaker_config):
        """Test initialization with configs."""
        manager = FaultToleranceManager(retry_config, circuit_breaker_config)

        assert manager.retry_config == retry_config
        assert manager.circuit_breaker_config == circuit_breaker_config


class TestExecuteWithRetry:
    """Test execute with retry."""

    def test_successful_execution(self, manager):
        """Test successful operation without retry."""
        operation = Mock(return_value="result")

        success, result, error = manager.execute_with_retry(1, operation)

        assert success is True
        assert result == "result"
        assert error is None

    def test_execution_with_single_failure_then_success(self, manager):
        """Test operation that fails once then succeeds."""
        operation = Mock(side_effect=[TimeoutError("timeout"), "result"])

        success, result, error = manager.execute_with_retry(1, operation)

        assert success is True
        assert result == "result"
        assert error is None

    def test_execution_with_all_failures(self, manager):
        """Test operation that fails all attempts."""
        operation = Mock(side_effect=TimeoutError("timeout"))

        success, result, error = manager.execute_with_retry(1, operation)

        assert success is False
        assert result is None
        assert error is not None

    def test_non_retryable_error(self, manager):
        """Test non-retryable error fails immediately."""
        operation = Mock(side_effect=ValueError("invalid"))

        success, result, error = manager.execute_with_retry(1, operation)

        assert success is False
        # Only called once (no retry)
        assert operation.call_count == 1

    def test_retry_backoff(self, manager):
        """Test retry uses backoff delays."""
        manager.retry_config.initial_delay_sec = 0.01
        operation = Mock(side_effect=TimeoutError("timeout"))

        start = time.time()
        success, result, error = manager.execute_with_retry(1, operation)
        elapsed = time.time() - start

        # Should take time due to delays
        assert elapsed > 0.01


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration."""

    def test_check_circuit_breaker_closed(self, manager):
        """Test circuit breaker check when closed."""
        assert manager.check_circuit_breaker(1)

    def test_check_circuit_breaker_open(self, manager, circuit_breaker_config):
        """Test circuit breaker check when open."""
        manager.circuit_breaker_config.failure_threshold = 2

        # Open the circuit
        manager.record_worker_failure(1)
        manager.record_worker_failure(1)

        assert not manager.check_circuit_breaker(1)

    def test_record_worker_success(self, manager):
        """Test recording worker success."""
        manager.record_worker_success(1)

        breaker = manager.get_circuit_breaker_status(1)[1]
        assert breaker.state == CircuitState.CLOSED

    def test_record_worker_failure(self, manager, circuit_breaker_config):
        """Test recording worker failure."""
        manager.circuit_breaker_config.failure_threshold = 2

        manager.record_worker_failure(1)
        manager.record_worker_failure(1)

        breaker = manager.get_circuit_breaker_status(1)[1]
        assert breaker.state == CircuitState.OPEN


class TestErrorLogging:
    """Test error logging."""

    def test_error_recorded(self, manager):
        """Test errors are recorded."""
        operation = Mock(side_effect=TimeoutError("timeout"))

        manager.execute_with_retry(1, operation)

        errors = manager.get_error_log()
        assert len(errors) > 0

    def test_error_has_details(self, manager):
        """Test error record has required details."""
        operation = Mock(side_effect=TimeoutError("timeout"))

        manager.execute_with_retry(1, operation)

        errors = manager.get_error_log()
        error = errors[0]

        assert error.worker_id == 1
        assert error.error_type == "TimeoutError"
        assert "timeout" in error.error_message.lower()

    def test_error_recovery_flag(self, manager):
        """Test error recovery flag is set."""
        operation = Mock(side_effect=TimeoutError("timeout"))

        manager.execute_with_retry(1, operation)

        errors = manager.get_error_log()
        # Timeout is retryable
        assert any(e.recoverable for e in errors)

    def test_filter_errors_by_worker(self, manager):
        """Test filtering errors by worker."""
        op1 = Mock(side_effect=TimeoutError())
        op2 = Mock(side_effect=TimeoutError())

        manager.execute_with_retry(1, op1)
        manager.execute_with_retry(2, op2)

        errors1 = manager.get_error_log(worker_id=1)
        errors2 = manager.get_error_log(worker_id=2)

        assert all(e.worker_id == 1 for e in errors1)
        assert all(e.worker_id == 2 for e in errors2)


class TestWorkerHealth:
    """Test worker health tracking."""

    def test_worker_health_initialized(self, manager):
        """Test worker health is initialized."""
        manager.record_worker_success(1)

        health = manager.get_worker_health(1)[1]
        assert health is not None

    def test_worker_success_count_increases(self, manager):
        """Test success count increases."""
        manager.record_worker_success(1)
        manager.record_worker_success(1)

        health = manager.get_worker_health(1)[1]
        assert health["success_count"] == 2

    def test_worker_failure_count_increases(self, manager):
        """Test failure count increases."""
        manager.record_worker_failure(1)
        manager.record_worker_failure(1)

        health = manager.get_worker_health(1)[1]
        assert health["failure_count"] == 2

    def test_multiple_workers_tracked(self, manager):
        """Test multiple workers can be tracked."""
        manager.record_worker_success(1)
        manager.record_worker_success(2)
        manager.record_worker_failure(3)

        health = manager.get_worker_health()
        assert len(health) == 3


class TestCircuitBreakerStatus:
    """Test circuit breaker status queries."""

    def test_get_all_circuit_breaker_status(self, manager):
        """Test getting all circuit breaker statuses."""
        manager.record_worker_success(1)
        manager.record_worker_success(2)

        status = manager.get_circuit_breaker_status()
        assert len(status) >= 2

    def test_get_single_circuit_breaker_status(self, manager):
        """Test getting single circuit breaker status."""
        manager.record_worker_success(1)

        status = manager.get_circuit_breaker_status(worker_id=1)
        assert 1 in status


class TestClearOperations:
    """Test clearing operations."""

    def test_clear_worker(self, manager):
        """Test clearing specific worker."""
        manager.record_worker_success(1)
        manager.record_worker_failure(1)

        manager.clear_worker(1)

        health = manager.get_worker_health(1)
        assert health[1] == {}

    def test_clear_all(self, manager):
        """Test clearing all data."""
        manager.record_worker_success(1)
        manager.record_worker_failure(2)

        operation = Mock(side_effect=TimeoutError())
        manager.execute_with_retry(3, operation)

        manager.clear_all()

        assert len(manager.get_error_log()) == 0
        assert len(manager.get_circuit_breaker_status()) == 0


class TestStatisticsComputation:
    """Test statistics computation."""

    def test_get_stats_empty(self, manager):
        """Test stats with no data."""
        stats = manager.get_stats()

        assert stats["total_errors"] == 0
        assert stats["workers_tracked"] == 0

    def test_get_stats_with_errors(self, manager):
        """Test stats with errors."""
        operation = Mock(side_effect=TimeoutError())

        manager.execute_with_retry(1, operation)

        stats = manager.get_stats()
        assert stats["total_errors"] > 0

    def test_get_stats_open_circuits(self, manager, circuit_breaker_config):
        """Test stats includes open circuits."""
        manager.circuit_breaker_config.failure_threshold = 1

        manager.record_worker_failure(1)

        stats = manager.get_stats()
        assert stats["open_circuits"] >= 1


class TestIntegration:
    """Integration tests."""

    def test_complete_fault_recovery_cycle(self, circuit_breaker_config):
        """Test complete fault recovery cycle."""
        manager = FaultToleranceManager(
            circuit_breaker_config=circuit_breaker_config
        )

        # Initial operation succeeds
        manager.record_worker_success(1)
        assert manager.check_circuit_breaker(1)

        # Multiple failures open circuit
        manager.circuit_breaker_config.failure_threshold = 2
        manager.record_worker_failure(1)
        manager.record_worker_failure(1)

        assert not manager.check_circuit_breaker(1)

        # After recovery timeout, circuit enters HALF_OPEN
        time.sleep(circuit_breaker_config.recovery_timeout_sec + 0.1)
        # Force state check
        breaker = manager._circuit_breakers[1]
        can_execute = breaker.can_execute()
        assert can_execute

        # Verify it's in HALF_OPEN
        assert breaker.get_state().state == CircuitState.HALF_OPEN

        # Successes close the circuit
        for _ in range(circuit_breaker_config.success_threshold):
            manager.record_worker_success(1)

        assert breaker.get_state().state == CircuitState.CLOSED

    def test_multiple_workers_independent(self, manager):
        """Test multiple workers fail independently."""
        operation1 = Mock(side_effect=TimeoutError())
        operation2 = Mock(return_value="success")

        # Worker 1 fails
        manager.execute_with_retry(1, operation1)

        # Worker 2 succeeds
        manager.execute_with_retry(2, operation2)

        # Worker 1 should have errors
        errors1 = manager.get_error_log(worker_id=1)
        assert len(errors1) > 0

        # Worker 2 should have no errors
        errors2 = manager.get_error_log(worker_id=2)
        assert len(errors2) == 0

    def test_retry_and_circuit_breaker_together(self, manager):
        """Test retry and circuit breaker work together."""
        attempt_count = [0]

        def operation():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise TimeoutError()
            return "success"

        # First attempt fails, retry succeeds
        success, result, error = manager.execute_with_retry(1, operation)

        assert success is True
        assert result == "success"


class TestEdgeCases:
    """Test edge cases."""

    def test_worker_id_zero(self, manager):
        """Test worker ID of zero."""
        manager.record_worker_success(0)

        health = manager.get_worker_health(0)
        assert health[0]["success_count"] == 1

    def test_large_worker_id(self, manager):
        """Test large worker ID."""
        worker_id = 999999

        manager.record_worker_success(worker_id)

        health = manager.get_worker_health(worker_id)
        assert worker_id in health

    def test_many_workers(self, manager):
        """Test tracking many workers."""
        for i in range(100):
            manager.record_worker_success(i)

        health = manager.get_worker_health()
        assert len(health) == 100
