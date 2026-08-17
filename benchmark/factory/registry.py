"""Implementation registry.

Maps string names (from config) to concrete implementation classes.
The orchestrator NEVER imports implementations directly — it uses the registry.
"""

from __future__ import annotations

from typing import Any

from benchmark.exceptions.memory_errors import RegistryResolutionError


class MemoryModuleRegistry:
    """Registry mapping config names to memory module implementation classes.

    Provides register/resolve pattern for dependency inversion.
    The orchestrator depends on this registry (an abstraction),
    never on concrete implementations.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: dict[str, type] = {}

    def register(self, name: str, implementation_class: type) -> None:
        """Register an implementation class under a config name.

        Args:
            name: The config name (e.g., "episodic_store", "preference_store").
            implementation_class: The concrete class to instantiate.

        Raises:
            RegistryResolutionError: If the name is already registered.
        """
        if name in self._registry:
            raise RegistryResolutionError(
                f"Module already registered: '{name}'. Existing: {self._registry[name].__name__}"
            )
        self._registry[name] = implementation_class

    def resolve(self, name: str, **constructor_kwargs: Any) -> Any:
        """Resolve a config name to an instantiated implementation.

        Args:
            name: The config name to resolve.
            **constructor_kwargs: Arguments to pass to the implementation constructor.

        Returns:
            An instantiated implementation object.

        Raises:
            RegistryResolutionError: If the name is not registered or instantiation fails.
        """
        if name not in self._registry:
            registered = ", ".join(sorted(self._registry.keys()))
            raise RegistryResolutionError(
                f"Unknown memory module: '{name}'. Registered: [{registered}]"
            )

        implementation_class = self._registry[name]

        try:
            return implementation_class(**constructor_kwargs)
        except TypeError as type_error:
            raise RegistryResolutionError(
                f"Failed to instantiate '{name}' ({implementation_class.__name__}): {type_error}"
            ) from type_error

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
            name: The config name to check.

        Returns:
            True if the name is registered.
        """
        return name in self._registry

    def registered_names(self) -> list[str]:
        """List all registered module names.

        Returns:
            Sorted list of registered names.
        """
        return sorted(self._registry.keys())

    def clear(self) -> None:
        """Remove all registrations. Used for testing."""
        self._registry.clear()


class ServiceRegistry:
    """Generic service registry for non-memory components (tokenizers, cost calculators, etc.).

    Designed to keep orchestrator free from concrete imports and to provide
    a single place to resolve auxiliary services.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(self, name: str, instance: Any) -> None:
        """Register an instance under a service name.

        Args:
            name: The service name (e.g., "tokenizer", "cost_tracker").
            instance: The concrete instance to register.

        Raises:
            RegistryResolutionError: If the service name is already registered.
        """
        if name in self._services:
            raise RegistryResolutionError(f"Service already registered: '{name}'")
        self._services[name] = instance

    def resolve(self, name: str) -> Any:
        """Resolve a service name to its registered instance.

        Args:
            name: The service name to resolve.

        Returns:
            The registered instance.

        Raises:
            RegistryResolutionError: If the service name is not registered.
        """
        if name not in self._services:
            registered = ", ".join(sorted(self._services.keys()))
            raise RegistryResolutionError(f"Unknown service: '{name}'. Registered: [{registered}]")
        return self._services[name]

    def registered_names(self) -> list[str]:
        """List all registered service names.

        Returns:
            Sorted list of registered service names.
        """
        return sorted(self._services.keys())

    def clear(self) -> None:
        """Remove all service registrations. Used for testing."""
        self._services.clear()


class RetrievalStrategyRegistry:
    """Registry for retrieval strategy implementations.

    Maps retrieval strategy names from config to concrete strategy classes.
    Examples: "bm25", "embeddings", "llm", "database", "hybrid"
    """

    def __init__(self) -> None:
        """Initialize an empty strategy registry."""
        self._strategies: dict[str, type] = {}

    def register(self, name: str, strategy_class: type) -> None:
        """Register a retrieval strategy.

        Args:
            name: Strategy name (e.g., "bm25", "embeddings").
            strategy_class: The concrete strategy class.

        Raises:
            RegistryResolutionError: If already registered.
        """
        if name in self._strategies:
            raise RegistryResolutionError(
                f"Strategy already registered: '{name}'. "
                f"Existing: {self._strategies[name].__name__}"
            )
        self._strategies[name] = strategy_class

    def resolve(self, name: str, **constructor_kwargs: Any) -> Any:
        """Resolve a strategy name to an instantiated strategy.

        Args:
            name: Strategy name to resolve.
            **constructor_kwargs: Arguments for strategy constructor.

        Returns:
            Instantiated strategy instance.

        Raises:
            RegistryResolutionError: If not registered or instantiation fails.
        """
        if name not in self._strategies:
            registered = ", ".join(sorted(self._strategies.keys()))
            raise RegistryResolutionError(
                f"Unknown retrieval strategy: '{name}'. Registered: [{registered}]"
            )

        strategy_class = self._strategies[name]

        try:
            return strategy_class(**constructor_kwargs)
        except TypeError as type_error:
            raise RegistryResolutionError(
                f"Failed to instantiate '{name}' ({strategy_class.__name__}): {type_error}"
            ) from type_error

    def is_registered(self, name: str) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Strategy name.

        Returns:
            True if registered.
        """
        return name in self._strategies

    def registered_names(self) -> list[str]:
        """List all registered strategy names.

        Returns:
            Sorted list of strategy names.
        """
        return sorted(self._strategies.keys())

    def clear(self) -> None:
        """Clear all strategies. Used for testing."""
        self._strategies.clear()
