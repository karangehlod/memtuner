"""Unit tests for time engine: TimeProvider interface and SimulatedClock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from benchmark.time.provider import TimeProvider
from benchmark.time.simulated_clock import SimulatedClock

FIXED_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.unit
class TestSimulatedClock:
    """Tests for the SimulatedClock implementation."""

    def test_is_time_provider(self) -> None:
        clock = SimulatedClock()
        assert isinstance(clock, TimeProvider)

    def test_starts_at_day_zero(self) -> None:
        clock = SimulatedClock(epoch=FIXED_EPOCH)
        assert clock.current_day() == 0

    def test_starts_at_epoch_timestamp(self) -> None:
        clock = SimulatedClock(epoch=FIXED_EPOCH)
        assert clock.current_timestamp() == FIXED_EPOCH

    def test_advance_day_increments(self) -> None:
        clock = SimulatedClock(epoch=FIXED_EPOCH)
        clock.advance_day()
        assert clock.current_day() == 1
        assert clock.current_timestamp() == FIXED_EPOCH + timedelta(days=1)

    def test_advance_multiple_days(self) -> None:
        clock = SimulatedClock(epoch=FIXED_EPOCH)
        for _ in range(7):
            clock.advance_day()
        assert clock.current_day() == 7
        assert clock.current_timestamp() == FIXED_EPOCH + timedelta(days=7)

    def test_reset_returns_to_day_zero(self) -> None:
        clock = SimulatedClock(epoch=FIXED_EPOCH)
        clock.advance_day()
        clock.advance_day()
        clock.reset()
        assert clock.current_day() == 0
        assert clock.current_timestamp() == FIXED_EPOCH

    def test_default_epoch_is_2026(self) -> None:
        clock = SimulatedClock()
        timestamp = clock.current_timestamp()
        assert timestamp.year == 2026
        assert timestamp.month == 1
        assert timestamp.day == 1

    def test_deterministic_across_instances(self) -> None:
        clock_a = SimulatedClock(epoch=FIXED_EPOCH)
        clock_b = SimulatedClock(epoch=FIXED_EPOCH)
        for _ in range(5):
            clock_a.advance_day()
            clock_b.advance_day()
        assert clock_a.current_day() == clock_b.current_day()
        assert clock_a.current_timestamp() == clock_b.current_timestamp()
