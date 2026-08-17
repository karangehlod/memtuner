# Phase 3 — Production-Grade Memory Benchmark

> Make this benchmark produce numbers that match or explain published baselines,
> and become a true tool for tuning decay/weightage parameters.

## The Three Additions

### A. Configurable Decay-in-Ranking

**Problem:** Decay currently doesn't affect retrieval — all λ values give the same recall.

**Solution:** Add a `decay_ranking_alpha` parameter:
```
rank_score = strategy_score × importance × (decay_factor ^ alpha)

alpha = 0.0 → current behavior (decay doesn't affect ranking)
alpha = 0.5 → moderate recency bias
alpha = 1.0 → full decay in ranking (newer memories ranked higher)
```

This lets users sweep alpha to find: "How much should recency matter vs relevance?"

**Open-source dependencies:** None (pure math).

---

### B. LLM-as-Judge Evaluation

**Problem:** We measure retrieval accuracy (ID matching), not answer quality.

**Solution:** After retrieval, generate an answer using a local LLM and score it.

**Pipeline:**
```
1. Retrieve top-K memories
2. Build prompt: "Given these memories, answer: {query}"
3. Call local LLM (via Ollama API) to generate answer
4. Score: compare generated answer vs gold answer
   - Exact match (simple)
   - Semantic similarity (embedding cosine)
   - LLM-as-judge (another LLM scores 1-5)
```

**Open-source models (via Ollama):**
- **Generation:** Llama 3.1 8B, Mistral 7B, Phi-3
- **Judge:** Flow Judge 3.8B (purpose-built for evaluation), or same model
- **Embeddings:** all-MiniLM-L6-v2 (already have it)

**Scoring methods (all local, no API keys):**
1. **F1 token overlap** — word-level overlap between generated and gold answer
2. **Semantic similarity** — cosine(embed(generated), embed(gold))
3. **LLM judge** — "Rate 1-5: Does this answer match the expected answer?"

---

### C. Better Retrieval (Session-Aware Embeddings)

**Problem:** BM25 gets 53% on LoCoMo, published systems get 88%+.

**Solution:** The gap is because published systems understand conversational structure:
- Session boundaries (which turns belong together)
- Speaker attribution (who said what)
- Temporal ordering (when was this said relative to other things)

**Approach:** Improve the embedding strategy to chunk by session (not individual turns):
```
Current:  each turn → separate embedding → independent retrieval
Improved: each session → combined embedding → contextual retrieval
          "Session 3 (Day 45): Alice said she prefers Postgres, discussed migration timeline"
```

**Open-source dependencies:** sentence-transformers (already have it).

---

## Implementation Order

1. **A (decay-in-ranking)** — 1 file change, immediately testable
2. **C (session-aware embeddings)** — improves numbers significantly
3. **B (LLM-as-judge)** — requires Ollama running, most complex

## Validation Targets

After all three:
- LoCoMo R@10 should reach 70-80% (from current 53-66%)
- LongMemEval should show meaningful decay effect
- LLM-judge scores should correlate with retrieval metrics
- Decay sweep should show actual recall differences

## Dependencies

| Feature | Requires | Open-source? | Local? |
|---------|----------|--------------|--------|
| A: Decay ranking | Nothing | ✅ | ✅ |
| B: LLM judge | Ollama + Llama3 | ✅ | ✅ |
| C: Session embeddings | sentence-transformers | ✅ | ✅ |

All run locally. No API keys. No cloud services. Fully reproducible.
