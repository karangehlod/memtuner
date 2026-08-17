from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Tokenizer interface for counting and encoding tokens deterministically.

    Implementations must be stateless and deterministic for the same input.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens for the provided text."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Return a deterministic sequence of token ids for the given text."""
