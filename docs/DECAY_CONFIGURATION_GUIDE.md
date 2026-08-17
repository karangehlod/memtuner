# Decay Factor Configuration Guide

## Dataset Overview

Your benchmark dataset spans **720 days** (exactly 2 years):
- **Start date:** 8 June 2024
- **End date:** 30 May 2026
- **Total span:** 720 days (24 months, ~2 years)

### Gold Memory Age Distribution

The memories that should be retrieved (gold labels) are distributed as follows:

| Metric | Value |
|--------|-------|
| Minimum age | 1 day |
| Maximum age | 294 days |
| **Average age** | **113.7 days** |
| Median age | ~100 days |
| % > 7 days | 92.3% |
| % > 30 days | 79.1% |
| **% > 100 days** | **51.3%** |

**Key insight:** Over half of your correct memories are 100+ days old. Aggressive decay will destroy retrieval performance.

---

## Exponential Decay Formula

The benchmark uses **exponential decay** by default:

$$\text{weight}(t) = e^{-\lambda \cdot t}$$

Where:
- **t** = age of memory in days (today = day 0, yesterday = day 1)
- **λ** (lambda) = decay rate parameter (0.0 = no decay, higher = faster decay)
- **weight** = multiplicative factor applied to memory's retrieval score before ranking

### Half-Life Interpretation

Another way to think about decay is via **half-life** — the age at which memories retain 50% of their original score:

$$\text{Half-life} = \frac{\ln(2)}{\lambda} = \frac{0.693}{\lambda} \text{ days}$$

---

## Decay Factor Selection Guide

### Scenario: 720-Day Dataset (Your Case)

For a 720-day corpus like yours, here are recommended λ values by use case:

#### **λ = 0.0 (No Decay) — "Information Preservation"**

Best for: **Long-term knowledge systems** where older memories are still valuable.

| Age | Weight | Interpretation |
|-----|--------|---|
| Today (0d) | 100.0% | Baseline |
| 15 days | 100.0% | No penalty |
| 30 days | 100.0% | No penalty |
| 90 days | 100.0% | Archival memories fully valued |
| 294 days | 100.0% | Very old memories still matter |

**Use case:** Customer support systems, medical histories, legal archives, long-term project memory.

**Trade-off:** Old stale information might persist. Mitigate with pruning (see section below).

---

#### **λ = 0.001 (Very Gentle Decay) — "Archival Protection"**

Best for: **Systems with predominantly old memories** where retrieval quality matters more than recency bias.

**Half-life: 693 days** (no meaningful decay within your 720-day dataset)

| Age | Weight | Half-Life Check |
|-----|--------|---|
| Today (0d) | 100.0% | Baseline |
| 15 days | 98.5% | Nearly no penalty |
| 30 days | 97.0% | Still near-full value |
| 90 days | 91.4% | Minor penalty for archival window |
| 113 days (avg) | 89.3% | ← **Your average gold memory** |
| 294 days (max) | 73.9% | Very old memories still weighted 74% |

**Benchmark finding:** At λ=0.001, the hybrid strategy achieves **recall=0.389**, functionally identical to no decay (recall=0.393). This is the gentlest useful decay.

**Use case:** Long-term memory systems, corporate knowledge bases, research archives.

---

#### **λ = 0.005 (Light Decay) — "Balanced History"**

Best for: **Mixed timeframes** where you want to preserve old important info but gradually fade noise.

**Half-life: 139 days** (decays by 50% after ~4.6 months)

| Age | Weight | Interpretation |
|-----|--------|---|
| Today (0d) | 100.0% | Baseline |
| 15 days | 92.7% | Slight discount |
| 30 days | 86.1% | Modest discount |
| 90 days | 60.7% | **Half-way to half-life** |
| 113 days | 57.1% | ← **Average gold memory loses 43%** |
| 139 days | 50.0% | Half-life point |
| 294 days | 12.4% | Very old memories heavily penalized |

**Benchmark finding:** At λ=0.005, hybrid achieves **recall=0.297** (24% drop from no decay).

**Use case:** Chat systems (1-2 month relevance window), task tracking, event-based memories.

---

#### **λ = 0.01 (Moderate Decay) — "Recency Bias"**

Best for: **Short-term focused systems** where old memories are mostly noise.

**Half-life: 69 days** (decays by 50% after ~2.3 months)

| Age | Weight | Interpretation |
|-----|--------|---|
| Today (0d) | 100.0% | Baseline |
| 15 days | 86.1% | Noticeable penalty |
| 30 days | 74.2% | 26% weight loss |
| 90 days | 40.7% | **Majority of weight gone** |
| 113 days | 31.9% | ← **Average gold memory loses 68%** |
| 139 days | 25.0% | Half-life point |
| 294 days | 5.2% | Ancient memories nearly ignored |

**Benchmark finding:** At λ=0.01, hybrid crashes to **recall=0.227** (42% drop from no decay). This is catastrophic for a long-term system.

**⚠️ CRITICAL:** For your 720-day dataset with 51% of gold > 100 days, λ=0.01 is **too aggressive**. You lose half the correct answers.

**Use case:** Short-term chat (Slack, Discord), daily standup notes, transient events.

---

#### **λ = 0.02 (Strong Decay) — "Fresh-First"**

Best for: **Systems where recency is paramount** and historical data is mostly obsolete.

**Half-life: 35 days** (decays by 50% after ~1.2 months)

| Age | Weight | Interpretation |
|-----|--------|---|
| Today (0d) | 100.0% | Baseline |
| 15 days | 74.2% | Major penalty |
| 30 days | 55.1% | **Majority weight gone** |
| 90 days | 16.5% | Nearly forgotten |
| 113 days | 10.2% | ← **Average gold memory loses 90%** |

**Benchmark finding:** λ=0.02 would drop recall further (estimated ~0.10-0.15).

**Use case:** Stock prices, weather, real-time feeds, highly ephemeral data.

---

#### **λ = 0.05 (Aggressive Decay) — "Ultra-Recency"**

Best for: **Real-time systems** where anything older than 2 weeks is irrelevant.

**Half-life: 14 days**

| Age | Weight | Interpretation |
|-----|--------|---|
| Today (0d) | 100.0% | Baseline |
| 7 days | 69.3% | One week = 31% penalty |
| 14 days | 48.0% | Half-life point |
| 30 days | 22.3% | Month-old data mostly ignored |
| 90 days | 0.1% | Effectively zero weight |

**Use case:** Stock tickers, live sports, breaking news.

---

## Decay Decision Matrix

| Dataset Span | Use Case | Recommended λ | Why |
|---|---|---|---|
| **720 days** (yours) | Long-term knowledge | **0.0 (none)** | Half your gold is 100+ days old |
| **720 days** (yours) | Balanced (preserve with noise control) | **0.001** | Very gentle, functionally ≈ none |
| **720 days** (yours) | Mixed, allow some old fade | **0.005** | 139d half-life, 43% weight loss at avg age |
| 90 days | Quarterly reviews, projects | **0.01** | 69d half-life, good for 3-month window |
| 30 days | Monthly standups | **0.02** | 35d half-life, old memories fade quickly |
| 7 days | Daily chat/events | **0.05** | 14d half-life, anything older is noise |

---

## Pruning: The Complementary Mechanism

Decay **multiplies** memory scores, but **pruning** **removes** memories entirely. They work together:

### How Pruning Works

1. **Calculate score** for each memory at query time:
   ```
   score = importance × decay_factor × retrieval_score
   ```
   where:
   - `importance` = inherent value (0.0–1.0, set during memory injection)
   - `decay_factor` = e^(-λ * age)
   - `retrieval_score` = BM25/embedding similarity

2. **Remove memories below threshold**:
   ```
   if score < pruning_threshold:
       delete memory
   ```

3. **Rank remaining memories** by score, retrieve top-k.

### Pruning Threshold Selection

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| **0.05** | Very aggressive pruning | Systems with many irrelevant memories |
| **0.10** | Moderate pruning | Typical long-term systems |
| **0.30** | Light pruning (default) | Preserve as much history as possible |
| **0.50** | Almost no pruning | Archives, compliance (never delete) |

### Interaction with Decay

- **If λ=0.0 (no decay):** Pruning removes only truly low-importance memories
- **If λ=0.001 (very gentle):** Pruning removes old low-importance + some borderline old memories
- **If λ=0.01 (moderate):** Pruning removes 60%+ of memories older than 90 days, regardless of importance
- **If λ=0.05 (aggressive):** Pruning combined with decay aggressively forgets everything older than 30 days

---

## Your Benchmark Results: The λ Cliff

Your empirical testing revealed a **cliff** in performance:

```
λ value   | Hybrid Recall | vs Baseline | Strategy Impact
----------|---------------|-------------|------------------
0.0 (none)      0.3934       Baseline        All strategies stable
0.001           0.3885       -1.2%           Negligible impact
0.005           0.2970       -24%            Light damage
0.010           0.2270       -42%            **Severe damage**
```

**Interpretation:**
- **λ ≤ 0.001**: Effectively no loss (weight > 89% even at 113 days)
- **λ = 0.005**: Noticeable degradation starts (43% weight loss at avg age)
- **λ ≥ 0.01**: Catastrophic for long-term systems (68% weight loss at avg age)

---

## Configuration Examples

### Example 1: Medical Records (Long-term Archival)

**Goal:** Retrieve patient history from years ago while avoiding outdated diagnoses.

```yaml
memory:
  long_term:
    episodic_store:
      decay:
        type: exponential
        lambda: 0.001      # Very gentle: 693-day half-life
      pruning:
        strategy: score_threshold
        threshold: 0.10    # Only remove truly low-importance
```

**Effect:** 100-day-old records retain 89% weight, 294-day-old records retain 74%.

---

### Example 2: Project Tracker (Quarterly Window)

**Goal:** Focus on active project memories (3 months), gradually forget completed projects.

```yaml
memory:
  long_term:
    episodic_store:
      decay:
        type: exponential
        lambda: 0.01       # 69-day half-life (good for 3-month window)
      pruning:
        strategy: score_threshold
        threshold: 0.20    # Remove old low-value items
```

**Effect:** 90-day-old project tasks retain 41% weight, 30-day-old retain 74%.

---

### Example 3: Chat/Messaging (Weekly Relevance)

**Goal:** Prioritize recent conversations but keep some older context.

```yaml
memory:
  long_term:
    episodic_store:
      decay:
        type: exponential
        lambda: 0.05       # 14-day half-life (short window)
      pruning:
        strategy: score_threshold
        threshold: 0.15    # Aggressive removal of old noise
```

**Effect:** 7-day-old messages retain 69% weight, 30-day-old retain 22%.

---

## Tiered Decay Strategy (New)

For **your specific dataset**, we've also implemented **tiered decay**:

```yaml
decay:
  type: tiered
  lambda: 0.01
```

**How it works:**
- **0–7 days (working memory):** weight = 1.0 (no decay)
- **7–90 days (episodic window):** weight = e^(-0.01 * (t-7)) (gentle fade)
- **90+ days (archival):** weight = 1.0 (full protection)

**Rationale:** 51% of your gold memories are 100+ days old. Tiered decay protects them while still applying decay to the 7–90 day window where memories transition from fresh to historical.

**Expected performance:** Should outperform exponential decay at λ=0.01 while being stricter than λ=0.001.

---

## Quick Reference: Weight at Your Average Gold Memory Age (113 days)

| λ | Weight @ 113d | Interpretation |
|---|---|---|
| 0.000 | 100.0% | No decay |
| 0.001 | 89.3% | Nearly as good as no decay |
| 0.003 | 71.0% | Light decay |
| 0.005 | 57.1% | Moderate decay (benchmark: 24% recall loss) |
| 0.007 | 45.2% | Significant decay |
| 0.009 | 36.0% | Heavy decay |
| 0.010 | 31.9% | **Benchmark: 42% recall loss** |
| 0.020 | 10.2% | Very aggressive |
| 0.050 | 0.1% | Ultra-aggressive |

---

## How to Use This in Your Benchmark

Run grid search with target λ values:

```bash
python3.13 grid_search.py \
  --lambda-min 0.001 \
  --lambda-max 0.010 \
  --lambda-step 0.001 \
  --output-dir results/decay_study
```

This produces λ = [0.001, 0.002, 0.003, ..., 0.010] → 10 cells per strategy.

Plot recall vs λ to find your optimal operating point.

---

## Summary: What λ Should You Use?

For **your 720-day dataset** with **51% of gold memories > 100 days old**:

| Decision | Recommended λ |
|----------|---|
| **Maximize retrieval quality** | **0.0** (none) |
| **Best balance for this dataset** | **0.001** |
| **Allow some age-based fade** | **0.005** |
| **Only if truly short-term focused** | **0.01** |
| **⚠️ Avoid for your dataset** | **≥0.02** |

**Your empirical finding:** λ=0.001 is functionally equivalent to no decay (0.3885 vs 0.3934 recall), confirming that very gentle decay respects your long-term memory distribution while making pruning slightly more effective.
