"""Unit tests for factory registry and config resolver."""

from __future__ import annotations

import pytest

from benchmark.config.schema import (
    BenchmarkConfig,
    MemoryConfig,
    MemorySelectionConfig,
)
from benchmark.exceptions.memory_errors import RegistryResolutionError
from benchmark.factory.registry import MemoryModuleRegistry
from benchmark.factory.resolver import ConfigResolver
from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.short_term.episodic_buffer import EpisodicBuffer


@pytest.mark.unit
class TestMemoryModuleRegistry:
    """Tests for the MemoryModuleRegistry."""

    def test_register_and_resolve(self) -> None:
        registry = MemoryModuleRegistry()
        registry.register("episodic_buffer", EpisodicBuffer)
        instance = registry.resolve("episodic_buffer")
        assert isinstance(instance, EpisodicBuffer)

    def test_resolve_with_kwargs(self) -> None:
        registry = MemoryModuleRegistry()
        registry.register("episodic_buffer", EpisodicBuffer)
        instance = registry.resolve("episodic_buffer", capacity=10)
        assert isinstance(instance, EpisodicBuffer)

    def test_duplicate_registration_raises(self) -> None:
        registry = MemoryModuleRegistry()
        registry.register("episodic_buffer", EpisodicBuffer)
        with pytest.raises(RegistryResolutionError, match="already registered"):
            registry.register("episodic_buffer", EpisodicBuffer)

    def test_resolve_unregistered_raises(self) -> None:
        registry = MemoryModuleRegistry()
        with pytest.raises(RegistryResolutionError, match="Unknown memory module"):
            registry.resolve("nonexistent")

    def test_is_registered_true(self) -> None:
        registry = MemoryModuleRegistry()
        registry.register("episodic_buffer", EpisodicBuffer)
        assert registry.is_registered("episodic_buffer") is True

    def test_is_registered_false(self) -> None:
        registry = MemoryModuleRegistry()
        assert registry.is_registered("nonexistent") is False

    def test_resolved_instance_implements_interfaces(self) -> None:
        registry = MemoryModuleRegistry()
        registry.register("episodic_buffer", EpisodicBuffer)
        instance = registry.resolve("episodic_buffer")
        assert isinstance(instance, MemoryWriter)
        assert isinstance(instance, MemoryReader)


@pytest.mark.unit
class TestConfigResolver:
    """Tests for the ConfigResolver."""

    def test_resolve_all_enabled_modules(
        self, populated_registry: MemoryModuleRegistry
    ) -> None:
        config = BenchmarkConfig(
            memory=MemoryConfig(
                enabled=MemorySelectionConfig(
                    short_term=["episodic_buffer"],
                    long_term=["episodic_store"],
                )
            )
        )
        resolver = ConfigResolver(populated_registry)
        modules = resolver.resolve_memory_modules(config)
        assert "episodic_buffer" in modules
        assert "episodic_store" in modules
        assert isinstance(modules["episodic_buffer"], EpisodicBuffer)
        assert isinstance(modules["episodic_store"], EpisodicStore)

    def test_validate_config_with_valid_modules(
        self, populated_registry: MemoryModuleRegistry
    ) -> None:
        config = BenchmarkConfig(
            memory=MemoryConfig(
                enabled=MemorySelectionConfig(
                    short_term=["episodic_buffer"],
                    long_term=["episodic_store"],
                )
            )
        )
        resolver = ConfigResolver(populated_registry)
        errors = resolver.validate_config_against_registry(config)
        assert errors == []

    def test_validate_config_with_unknown_module(
        self, populated_registry: MemoryModuleRegistry
    ) -> None:
        config = BenchmarkConfig(
            memory=MemoryConfig(
                enabled=MemorySelectionConfig(
                    short_term=["nonexistent_module"],
                    long_term=[],
                )
            )
        )
        resolver = ConfigResolver(populated_registry)
        errors = resolver.validate_config_against_registry(config)
        assert len(errors) == 1
        assert "nonexistent_module" in errors[0]

    def test_resolve_empty_modules(
        self, populated_registry: MemoryModuleRegistry
    ) -> None:
        config = BenchmarkConfig(
            memory=MemoryConfig(
                enabled=MemorySelectionConfig(short_term=[], long_term=[])
            )
        )
        resolver = ConfigResolver(populated_registry)
        modules = resolver.resolve_memory_modules(config)
        assert modules == {}
