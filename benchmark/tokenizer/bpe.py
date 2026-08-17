from __future__ import annotations

from benchmark.tokenizer.interface import Tokenizer


class SimpleBPETokenizer(Tokenizer):
    """A very small, deterministic BPE-like tokenizer for testing.

    This is intentionally simple: it splits on whitespace and punctuation,
    lowercases, and then maps subword tokens to stable ids via a hash
    function. It is NOT intended as a production tokenizer but provides
    deterministic token counts for benchmark cost calculations and tests.
    """

    def __init__(self) -> None:
        # small static vocabulary for deterministic mapping (could be extended)
        self._vocab_seed = 123456789

    def _tokenize(self, text: str) -> list[str]:
        # naive split; keep letters and digits grouped
        import re

        tokens = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text.lower())
        # combine punctuation with surrounding token where appropriate
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        ids: list[int] = [self._stable_hash(t) for t in tokens]
        return ids

    def count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def _stable_hash(self, token: str) -> int:
        # deterministic mapping to a 32-bit id
        h = 2166136261
        for c in token:
            h = (h ^ ord(c)) * 16777619 & 0xFFFFFFFF
        return h
