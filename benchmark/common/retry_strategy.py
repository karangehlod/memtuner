"""Retry strategy with exponential backoff for transient errors.

Provides configurable retry logic for handling transient failures in I/O operations.
"""

import logging

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


class RetryStrategy:
    """Exponential backoff retry logic with configurable limits.

    Handles transient errors (network timeouts, connection errors) with
    exponential backoff delays. Non-transient errors fail immediately.

    Usage:
        retry = RetryStrategy(max_attempts=3, base_delay=1.0)
        while retry.attempts < retry.max_attempts:
            try:
                result = operation()
                return result
            except Exception as exc:
                if not retry.should_retry(exc):
                    return default_value
                delay = retry.get_delay()
                time.sleep(delay)
    """

    # Transient errors that can be retried
    TRANSIENT_EXCEPTIONS = (
        TimeoutError,
        ConnectionError,
        OSError,
        EOFError,
    )

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        """Initialize retry strategy.

        Args:
            max_attempts: Maximum number of retry attempts (including first).
            base_delay: Initial delay in seconds between retries.
            max_delay: Maximum delay cap in seconds.
        """
        self.max_attempts = max(1, max_attempts)
        self.base_delay = max(0.1, base_delay)
        self.max_delay = max(self.base_delay, max_delay)
        self.attempts = 0

    def _compute_delay(self) -> float:
        """Compute the backoff delay for the current attempt without advancing state."""
        return min(self.base_delay * (2 ** self.attempts), self.max_delay)

    def get_delay(self) -> float:
        """Return delay for the current attempt and advance the attempt counter.

        Delay grows as: base_delay * (2^attempts), capped at max_delay.
        Side effect: increments self.attempts.

        Returns:
            Delay in seconds.
        """
        delay = self._compute_delay()
        self.attempts += 1
        return delay

    def should_retry(self, exc: Exception) -> bool:
        """Determine if exception is transient and retryable.

        Handles both standard Python exceptions and httpx exceptions.

        Args:
            exc: The exception to evaluate.

        Returns:
            True if exception is transient and can be retried.
        """
        # Check standard transient exceptions
        if isinstance(exc, self.TRANSIENT_EXCEPTIONS):
            return True

        # Check httpx exceptions if available
        if httpx:
            httpx_transient = []
            # Add TimeoutException if available
            if hasattr(httpx, "TimeoutException"):
                httpx_transient.append(httpx.TimeoutException)
            # Add other common httpx exceptions
            if hasattr(httpx, "ConnectError"):
                httpx_transient.append(httpx.ConnectError)
            if hasattr(httpx, "ReadTimeout"):
                httpx_transient.append(httpx.ReadTimeout)
            if hasattr(httpx, "WriteTimeout"):
                httpx_transient.append(httpx.WriteTimeout)
            if hasattr(httpx, "PoolTimeout"):
                httpx_transient.append(httpx.PoolTimeout)

            if httpx_transient and isinstance(exc, tuple(httpx_transient)):
                return True

            # HTTP 5xx errors are transient, 4xx are not
            if hasattr(httpx, "HTTPStatusError") and isinstance(
                exc, httpx.HTTPStatusError
            ):
                return exc.response.status_code >= 500

        return False

    def can_retry(self) -> bool:
        """Check if retry attempts remain.

        Returns:
            True if more retry attempts are available.
        """
        return self.attempts < self.max_attempts

    def reset(self) -> None:
        """Reset attempt counter for a new retry sequence."""
        self.attempts = 0

    def __repr__(self) -> str:
        """String representation of retry strategy."""
        return (
            f"RetryStrategy(max_attempts={self.max_attempts}, "
            f"base_delay={self.base_delay}, max_delay={self.max_delay}, "
            f"attempts={self.attempts})"
        )
