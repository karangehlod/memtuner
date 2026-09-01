"""Manage fault tolerance and error recovery in distributed execution."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    initial_delay_sec: float = 1.0
    max_delay_sec: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1  # Random jitter to avoid thundering herd


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout_sec: float = 30.0  # Time in OPEN state before HALF_OPEN
    success_threshold: int = 2  # Successes in HALF_OPEN before closing


@dataclass
class ErrorRecord:
    """Record of an error that occurred."""
    timestamp: float
    worker_id: int | None
    error_type: str
    error_message: str
    attempt: int
    recoverable: bool


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    worker_id: int
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_state_change: float


class RetryStrategy:
    """Exponential backoff retry logic."""

    def __init__(self, config: RetryConfig | None = None):
        """Initialize retry strategy.

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self.attempt = 0

    def should_retry(self, exc: Exception) -> bool:
        """Determine if exception is retryable.

        Args:
            exc: Exception to check

        Returns:
            True if should retry
        """
        # Retryable errors
        retryable_types = (
            TimeoutError,
            ConnectionError,
            OSError,
            IOError,
        )

        return isinstance(exc, retryable_types)

    def get_backoff_delay(self) -> float:
        """Get delay for next retry attempt.

        Returns:
            Delay in seconds
        """
        import random

        # Exponential backoff: initial_delay * (multiplier ^ attempt)
        delay = min(
            self.config.initial_delay_sec * (self.config.backoff_multiplier ** self.attempt),
            self.config.max_delay_sec,
        )

        # Add jitter to prevent thundering herd
        jitter = delay * self.config.jitter_factor * random.random()
        delay_with_jitter = delay + jitter

        self.attempt += 1

        return delay_with_jitter

    def reset(self) -> None:
        """Reset attempt counter."""
        self.attempt = 0

    def is_exhausted(self) -> bool:
        """Check if retry attempts exhausted.

        Returns:
            True if max attempts exceeded
        """
        return self.attempt >= self.config.max_attempts


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, worker_id: int, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker.

        Args:
            worker_id: Worker identifier
            config: Circuit breaker configuration
        """
        self.worker_id = worker_id
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState(
            worker_id=worker_id,
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=0,
            last_failure_time=None,
            last_state_change=time.time(),
        )

    def can_execute(self) -> bool:
        """Check if operation can proceed.

        Returns:
            True if circuit allows execution
        """
        if self.state.state == CircuitState.CLOSED:
            return True

        if self.state.state == CircuitState.HALF_OPEN:
            return True

        # OPEN state
        if self.state.state == CircuitState.OPEN:
            elapsed = time.time() - self.state.last_state_change
            if elapsed >= self.config.recovery_timeout_sec:
                # Transition to HALF_OPEN to test recovery
                self._transition_to_half_open()
                return True

            return False

        return False

    def record_success(self) -> None:
        """Record successful operation."""
        self.state.failure_count = 0

        if self.state.state == CircuitState.HALF_OPEN:
            self.state.success_count += 1

            if self.state.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state.state == CircuitState.CLOSED:
            # Already closed, just continue
            pass

    def record_failure(self) -> None:
        """Record failed operation."""
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()
        self.state.success_count = 0

        if self.state.state == CircuitState.CLOSED:
            if self.state.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state.state == CircuitState.HALF_OPEN:
            # Failed while testing, go back to OPEN
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state.state = CircuitState.OPEN
        self.state.last_state_change = time.time()
        logger.warning(f"Circuit breaker OPEN for worker {self.worker_id}")

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state.state = CircuitState.HALF_OPEN
        self.state.success_count = 0
        self.state.failure_count = 0
        self.state.last_state_change = time.time()
        logger.info(f"Circuit breaker HALF_OPEN for worker {self.worker_id}")

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state.state = CircuitState.CLOSED
        self.state.failure_count = 0
        self.state.success_count = 0
        self.state.last_state_change = time.time()
        logger.info(f"Circuit breaker CLOSED for worker {self.worker_id}")

    def get_state(self) -> CircuitBreakerState:
        """Get current state.

        Returns:
            Current circuit breaker state
        """
        return self.state


class FaultToleranceManager:
    """Manage fault tolerance for distributed execution."""

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """Initialize fault tolerance manager.

        Args:
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

        self._error_log: list[ErrorRecord] = []
        self._circuit_breakers: dict[int, CircuitBreaker] = {}
        self._retry_strategies: dict[int, RetryStrategy] = {}
        self._worker_health: dict[int, dict[str, Any]] = {}

    def execute_with_retry(
        self,
        worker_id: int,
        operation: Callable[[], Any],
        operation_name: str = "operation",
    ) -> tuple[bool, Any, str | None]:
        """Execute operation with retry logic.

        Args:
            worker_id: Worker identifier
            operation: Callable to execute
            operation_name: Name for logging

        Returns:
            Tuple of (success, result, error_message)
        """
        retry_strategy = self._get_retry_strategy(worker_id)

        while not retry_strategy.is_exhausted():
            try:
                result = operation()
                self._record_success(worker_id)
                return True, result, None

            except Exception as exc:
                error_type = type(exc).__name__
                error_msg = str(exc)

                if not retry_strategy.should_retry(exc):
                    logger.error(
                        f"Non-retryable error in {operation_name} "
                        f"(worker {worker_id}): {error_type}: {error_msg}"
                    )
                    self._record_error(
                        worker_id,
                        error_type,
                        error_msg,
                        retry_strategy.attempt,
                        recoverable=False,
                    )
                    return False, None, error_msg

                # Retryable error
                if retry_strategy.is_exhausted():
                    logger.error(
                        f"Max retries exhausted for {operation_name} "
                        f"(worker {worker_id})"
                    )
                    self._record_error(
                        worker_id,
                        error_type,
                        error_msg,
                        retry_strategy.attempt,
                        recoverable=True,
                    )
                    return False, None, error_msg

                delay = retry_strategy.get_backoff_delay()
                logger.warning(
                    f"Retryable error in {operation_name} (worker {worker_id}), "
                    f"retry in {delay:.1f}s: {error_type}: {error_msg}"
                )

                self._record_error(
                    worker_id,
                    error_type,
                    error_msg,
                    retry_strategy.attempt,
                    recoverable=True,
                )

                time.sleep(delay)

        return False, None, "Max retries exhausted"

    def check_circuit_breaker(self, worker_id: int) -> bool:
        """Check if worker circuit is open.

        Args:
            worker_id: Worker identifier

        Returns:
            True if can execute (circuit not open)
        """
        breaker = self._get_circuit_breaker(worker_id)
        return breaker.can_execute()

    def record_worker_success(self, worker_id: int) -> None:
        """Record successful worker operation.

        Args:
            worker_id: Worker identifier
        """
        breaker = self._get_circuit_breaker(worker_id)
        breaker.record_success()
        self._update_worker_health(worker_id, "success")

    def record_worker_failure(self, worker_id: int) -> None:
        """Record failed worker operation.

        Args:
            worker_id: Worker identifier
        """
        breaker = self._get_circuit_breaker(worker_id)
        breaker.record_failure()
        self._update_worker_health(worker_id, "failure")

    def get_error_log(
        self,
        worker_id: int | None = None,
    ) -> list[ErrorRecord]:
        """Get error log.

        Args:
            worker_id: Optional filter by worker

        Returns:
            List of error records
        """
        if worker_id is None:
            return self._error_log.copy()

        return [e for e in self._error_log if e.worker_id == worker_id]

    def get_circuit_breaker_status(
        self,
        worker_id: int | None = None,
    ) -> dict[int, CircuitBreakerState]:
        """Get circuit breaker status.

        Args:
            worker_id: Optional filter by worker

        Returns:
            Dict mapping worker_id to circuit breaker state
        """
        if worker_id is not None:
            breaker = self._get_circuit_breaker(worker_id)
            return {worker_id: breaker.get_state()}

        return {wid: breaker.get_state() for wid, breaker in self._circuit_breakers.items()}

    def get_worker_health(
        self,
        worker_id: int | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Get worker health status.

        Args:
            worker_id: Optional filter by worker

        Returns:
            Dict mapping worker_id to health info
        """
        if worker_id is not None:
            return {worker_id: self._worker_health.get(worker_id, {})}

        return self._worker_health.copy()

    def clear_worker(self, worker_id: int) -> None:
        """Clear data for worker.

        Args:
            worker_id: Worker identifier
        """
        self._circuit_breakers.pop(worker_id, None)
        self._retry_strategies.pop(worker_id, None)
        self._worker_health.pop(worker_id, None)

    def clear_all(self) -> None:
        """Clear all tracking data."""
        self._error_log.clear()
        self._circuit_breakers.clear()
        self._retry_strategies.clear()
        self._worker_health.clear()

    # Private helper methods

    def _get_circuit_breaker(self, worker_id: int) -> CircuitBreaker:
        """Get or create circuit breaker for worker."""
        if worker_id not in self._circuit_breakers:
            self._circuit_breakers[worker_id] = CircuitBreaker(
                worker_id,
                self.circuit_breaker_config,
            )

        return self._circuit_breakers[worker_id]

    def _get_retry_strategy(self, worker_id: int) -> RetryStrategy:
        """Get or create retry strategy for worker."""
        if worker_id not in self._retry_strategies:
            self._retry_strategies[worker_id] = RetryStrategy(self.retry_config)

        # Reset for new attempt
        strategy = self._retry_strategies[worker_id]
        strategy.reset()
        return strategy

    def _record_error(
        self,
        worker_id: int | None,
        error_type: str,
        error_message: str,
        attempt: int,
        recoverable: bool,
    ) -> None:
        """Record error in log."""
        error_record = ErrorRecord(
            timestamp=time.time(),
            worker_id=worker_id,
            error_type=error_type,
            error_message=error_message,
            attempt=attempt,
            recoverable=recoverable,
        )

        self._error_log.append(error_record)

    def _record_success(self, worker_id: int) -> None:
        """Record worker success."""
        self.record_worker_success(worker_id)

    def _update_worker_health(self, worker_id: int, status: str) -> None:
        """Update worker health status."""
        if worker_id not in self._worker_health:
            self._worker_health[worker_id] = {
                "success_count": 0,
                "failure_count": 0,
                "last_status": None,
            }

        health = self._worker_health[worker_id]

        if status == "success":
            health["success_count"] = health.get("success_count", 0) + 1
        elif status == "failure":
            health["failure_count"] = health.get("failure_count", 0) + 1

        health["last_status"] = status
        health["last_update"] = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Get fault tolerance statistics.

        Returns:
            Stats dict
        """
        total_errors = len(self._error_log)
        recoverable_errors = sum(1 for e in self._error_log if e.recoverable)
        non_recoverable_errors = total_errors - recoverable_errors

        open_circuits = sum(
            1
            for b in self._circuit_breakers.values()
            if b.get_state().state == CircuitState.OPEN
        )

        return {
            "total_errors": total_errors,
            "recoverable_errors": recoverable_errors,
            "non_recoverable_errors": non_recoverable_errors,
            "open_circuits": open_circuits,
            "workers_tracked": len(self._worker_health),
        }
