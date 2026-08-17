"""Pack Registry — Discovers and resolves benchmark packs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.packs.base import BenchmarkPack


_REGISTRY: dict[str, type[BenchmarkPack]] = {}


def register_pack(name: str):
    """Decorator to register a benchmark pack class."""

    def decorator(cls: type[BenchmarkPack]):
        _REGISTRY[name] = cls
        return cls

    return decorator


class PackRegistry:
    """Registry for discovering and resolving benchmark packs."""

    @staticmethod
    def available() -> list[str]:
        """Return list of available pack names."""
        return list(_REGISTRY.keys())

    @staticmethod
    def get(name: str) -> BenchmarkPack:
        """Resolve and instantiate a pack by name.

        Args:
            name: Pack identifier (e.g., 'longmemeval', 'locomo', 'private').

        Returns:
            Instantiated BenchmarkPack.

        Raises:
            KeyError: If pack name is not registered.
        """
        if name not in _REGISTRY:
            available = ", ".join(_REGISTRY.keys())
            raise KeyError(f"Unknown pack '{name}'. Available: {available}")
        return _REGISTRY[name]()

    @staticmethod
    def register(name: str, pack_class: type[BenchmarkPack]) -> None:
        """Manually register a pack class."""
        _REGISTRY[name] = pack_class


# Import packs to trigger registration
def _auto_register():
    """Import all pack modules to trigger @register_pack decorators."""
    import benchmark.packs.locomo.adapter
    import benchmark.packs.longmemeval.adapter
    import benchmark.packs.private.adapter  # noqa: F401


_auto_register()
