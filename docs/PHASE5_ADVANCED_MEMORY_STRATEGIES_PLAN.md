# Phase 5: Advanced Memory Strategies Integration

**Objective:** Integrate retrieval benchmarking with advanced memory management strategies to create a comprehensive agentic memory system.

**Estimated Duration:** 40 hours (5 weeks part-time) | **Target Completion:** September 2026

---

## Executive Summary

Phase 5 builds on Phase 4's retrieval framework to implement **5 advanced memory strategies** that leverage retrieval for intelligent memory management:

1. **Semantic Memory Clustering** - Group similar memories for efficient retrieval
2. **Attention-Based Retrieval** - Dynamically weight memory importance
3. **Hierarchical Memory Consolidation** - Transfer memories between short/long-term
4. **Query-Adaptive Strategy Selection** - Choose retriever based on query type
5. **Personalized Memory Ranking** - Rank memories by user/agent preferences

---

## Architecture Overview

### Memory System Stack

```
┌─────────────────────────────────────────────────────┐
│         Phase 5: Advanced Memory Strategies         │
├─────────────────────────────────────────────────────┤
│  - Semantic Clustering       - Attention Weighting   │
│  - Consolidation             - Strategy Selection    │
│  - Personalized Ranking      - Dynamic Fusion       │
├─────────────────────────────────────────────────────┤
│     Phase 4: Retrieval Benchmark (11 strategies)    │
├─────────────────────────────────────────────────────┤
│   Phase 3: Dataset Framework (14 datasets)          │
├─────────────────────────────────────────────────────┤
│   Phase 2: Memory Storage & Management              │
├─────────────────────────────────────────────────────┤
│   Phase 1: Core Foundation & Infrastructure         │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
Query (from Agent)
  ↓
[Phase 5] Query Analysis & Strategy Selection
  ↓
[Phase 4] Retrieve Candidates (chosen strategy)
  ↓
[Phase 5] Attention-Based Re-weighting
  ↓
[Phase 5] Personalized Ranking
  ↓
[Phase 5] Consolidation Decision (promote/demote memories)
  ↓
Ranked Results (to Agent) + State Updates
```

---

## Task Breakdown

### T5.1: Semantic Memory Clustering (8 hours)

**Goal:** Group similar memories for efficient retrieval and conceptual organization.

#### Implementation:

**SemanticClusterManager** (new class)
```python
class SemanticClusterManager:
    def __init__(self, embedding_adapter, num_clusters=50):
        """Initialize clustering with dense embeddings."""
        self.embedding_adapter = embedding_adapter
        self.num_clusters = num_clusters
        self.clusters = {}
        self.memory_to_cluster = {}
    
    def cluster_memories(self, memories: list[Memory]) -> dict[int, list[Memory]]:
        """Cluster memories by semantic similarity using K-means."""
        # 1. Embed all memories
        embeddings = self.embedding_adapter.encode([m.content for m in memories])
        
        # 2. K-means clustering
        clusters = self._kmeans(embeddings, k=self.num_clusters)
        
        # 3. Organize by cluster
        clustered = {}
        for cluster_id, indices in clusters.items():
            clustered[cluster_id] = [memories[i] for i in indices]
            for idx in indices:
                self.memory_to_cluster[memories[idx].id] = cluster_id
        
        return clustered
    
    def get_cluster_summary(self, cluster_id: int) -> str:
        """Generate summary of cluster themes."""
        cluster_mems = self.clusters.get(cluster_id, [])
        # Use LLM to summarize cluster theme
        theme = self._summarize_theme([m.content for m in cluster_mems])
        return theme
    
    def find_similar_cluster(self, query: str) -> int:
        """Find most relevant cluster for query."""
        query_embedding = self.embedding_adapter.encode([query])[0]
        cluster_centers = self._compute_cluster_centers()
        best_cluster = max(
            cluster_centers.items(),
            key=lambda x: cosine_similarity(query_embedding, x[1])
        )[0]
        return best_cluster
    
    def merge_clusters(self, cluster_ids: list[int]) -> int:
        """Merge multiple clusters into one."""
        new_cluster_id = max(self.clusters.keys()) + 1
        merged_memories = []
        for cid in cluster_ids:
            merged_memories.extend(self.clusters[cid])
        self.clusters[new_cluster_id] = merged_memories
        return new_cluster_id
    
    def split_cluster(self, cluster_id: int, num_subgroups: int) -> list[int]:
        """Split a cluster into subclusters."""
        cluster_mems = self.clusters[cluster_id]
        # Re-cluster into subclusters
        sub_ids = []
        for i in range(num_subgroups):
            new_id = max(self.clusters.keys()) + 1
            self.clusters[new_id] = []
            sub_ids.append(new_id)
        return sub_ids
```

**Tests:**
- Clustering accuracy (silhouette score > 0.6)
- Cluster summarization quality
- Query-to-cluster relevance
- Cluster merging/splitting correctness

**Metrics:**
- Silhouette coefficient (cluster quality)
- Query-cluster MRR (retrieval effectiveness)
- Cluster coherence (topic consistency)

---

### T5.2: Attention-Based Memory Weighting (8 hours)

**Goal:** Dynamically assign importance scores to memories based on relevance, recency, and frequency.

#### Implementation:

**AttentionWeighter** (new class)
```python
class AttentionWeighter:
    def __init__(self, query_encoder, memory_encoder):
        """Initialize attention mechanism."""
        self.query_encoder = query_encoder
        self.memory_encoder = memory_encoder
        self.attention_history = {}
    
    def compute_attention_scores(
        self,
        query: str,
        candidate_memories: list[Memory],
        weights: dict = None,  # Weights for different factors
    ) -> dict[str, float]:
        """Compute attention scores using multi-factor weighting."""
        scores = {}
        
        # Default weights if not specified
        if weights is None:
            weights = {
                "semantic_relevance": 0.40,
                "recency": 0.25,
                "frequency": 0.20,
                "coherence": 0.15,
            }
        
        # 1. Semantic relevance (query-memory similarity)
        query_emb = self.query_encoder.encode([query])[0]
        for mem in candidate_memories:
            mem_emb = self.memory_encoder.encode([mem.content])[0]
            relevance = cosine_similarity(query_emb, mem_emb)
            
            # 2. Recency (exponential decay)
            recency = math.exp(-decay_rate * (now - mem.last_accessed))
            
            # 3. Frequency (log-scaled)
            frequency = math.log1p(mem.access_count)
            
            # 4. Coherence (how well fits with query context)
            coherence = self._coherence_with_context(query, mem.content)
            
            # Weighted combination
            scores[mem.id] = (
                weights["semantic_relevance"] * relevance +
                weights["recency"] * recency +
                weights["frequency"] * frequency +
                weights["coherence"] * coherence
            )
            
            # Track attention
            self.attention_history[mem.id].append({
                "query": query,
                "score": scores[mem.id],
                "timestamp": now,
            })
        
        return scores
    
    def adaptive_weights(self, query_type: str) -> dict[str, float]:
        """Adjust weights based on query type."""
        query_types = {
            "factual": {"semantic": 0.60, "recency": 0.10, "frequency": 0.20, "coherence": 0.10},
            "temporal": {"semantic": 0.30, "recency": 0.50, "frequency": 0.10, "coherence": 0.10},
            "creative": {"semantic": 0.40, "recency": 0.10, "frequency": 0.10, "coherence": 0.40},
            "analytical": {"semantic": 0.45, "recency": 0.15, "frequency": 0.25, "coherence": 0.15},
        }
        return query_types.get(query_type, query_types["analytical"])
```

**Tests:**
- Attention score range [0, 1]
- Recency decay correctness
- Frequency scaling
- Coherence calculation
- Query-type specific weighting

**Metrics:**
- Weight sum = 1.0 (normalized)
- Attention stability (consistency)
- Correlation with relevance judgments

---

### T5.3: Hierarchical Memory Consolidation (10 hours)

**Goal:** Manage memory lifecycle: working → episodic → semantic consolidation.

#### Implementation:

**MemoryConsolidationEngine** (new class)
```python
class MemoryConsolidationEngine:
    def __init__(self):
        """Initialize consolidation with three memory tiers."""
        self.working_memory = []      # Recent, volatile
        self.episodic_memory = []     # Medium-term, event-based
        self.semantic_memory = []     # Long-term, knowledge
        
        self.consolidation_policy = {
            "working_to_episodic": {
                "age_threshold": 3600,  # seconds
                "access_count_min": 2,
            },
            "episodic_to_semantic": {
                "age_threshold": 86400,  # 1 day
                "access_count_min": 5,
                "relevance_threshold": 0.7,
            },
        }
    
    def consolidate_working_to_episodic(self) -> list[Memory]:
        """Promote working memories to episodic if criteria met."""
        promoted = []
        remaining_working = []
        
        for mem in self.working_memory:
            age_seconds = (time.time() - mem.created_at)
            
            if (age_seconds > self.consolidation_policy["working_to_episodic"]["age_threshold"] and
                mem.access_count >= self.consolidation_policy["working_to_episodic"]["access_count_min"]):
                
                # Apply episodic consolidation
                consolidated = self._episodic_consolidate(mem)
                self.episodic_memory.append(consolidated)
                promoted.append(consolidated)
            else:
                remaining_working.append(mem)
        
        self.working_memory = remaining_working
        return promoted
    
    def consolidate_episodic_to_semantic(self) -> list[Memory]:
        """Promote episodic memories to semantic if criteria met."""
        promoted = []
        remaining_episodic = []
        
        for mem in self.episodic_memory:
            age_seconds = (time.time() - mem.created_at)
            
            if (age_seconds > self.consolidation_policy["episodic_to_semantic"]["age_threshold"] and
                mem.access_count >= self.consolidation_policy["episodic_to_semantic"]["access_count_min"] and
                mem.relevance >= self.consolidation_policy["episodic_to_semantic"]["relevance_threshold"]):
                
                # Apply semantic consolidation (abstraction)
                consolidated = self._semantic_consolidate(mem)
                self.semantic_memory.append(consolidated)
                promoted.append(consolidated)
            else:
                remaining_episodic.append(mem)
        
        self.episodic_memory = remaining_episodic
        return promoted
    
    def decay_working_memory(self):
        """Remove old, unused working memories."""
        cutoff = time.time() - 7200  # 2 hours
        self.working_memory = [
            m for m in self.working_memory 
            if m.last_accessed > cutoff or m.access_count > 0
        ]
    
    def _episodic_consolidate(self, mem: Memory) -> Memory:
        """Episodic consolidation: Add context, metadata."""
        return Memory(
            id=mem.id,
            content=mem.content,
            tier="episodic",
            context=self._extract_context(mem),
            metadata={
                "original_tier": "working",
                "consolidation_time": time.time(),
            }
        )
    
    def _semantic_consolidate(self, mem: Memory) -> Memory:
        """Semantic consolidation: Abstraction and generalization."""
        generalized = self._generalize_content(mem.content)
        return Memory(
            id=mem.id,
            content=generalized,
            tier="semantic",
            abstract=True,
            metadata={
                "original_tier": "episodic",
                "consolidation_time": time.time(),
            }
        )
    
    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "working": len(self.working_memory),
            "episodic": len(self.episodic_memory),
            "semantic": len(self.semantic_memory),
            "total": len(self.working_memory) + len(self.episodic_memory) + len(self.semantic_memory),
        }
```

**Tests:**
- Consolidation criteria correctly applied
- Memory promotion/demotion
- Decay of old memories
- Multi-tier consistency
- Statistics accuracy

**Metrics:**
- Promotion rate (memories moving up)
- Decay rate (memories removed)
- Tier distribution (how memories distributed)
- Consolidation overhead

---

### T5.4: Query-Adaptive Strategy Selection (6 hours)

**Goal:** Automatically choose best retriever based on query characteristics.

#### Implementation:

**AdaptiveStrategySelector** (new class)
```python
class AdaptiveStrategySelector:
    def __init__(self, retrieval_registry):
        """Initialize with access to all retrieval strategies."""
        self.registry = retrieval_registry
        self.strategy_performance = {}
        self.query_history = []
    
    def analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze query to determine characteristics."""
        return {
            "length": len(query.split()),
            "entity_count": self._count_entities(query),
            "temporal": self._has_temporal_terms(query),
            "boolean": self._has_boolean_operators(query),
            "semantic": self._has_semantic_terms(query),
            "type": self._classify_query_type(query),
        }
    
    def select_strategy(
        self,
        query: str,
        constraints: dict = None,  # Max latency, memory, etc.
    ) -> str:
        """Select best strategy for query."""
        analysis = self.analyze_query(query)
        
        # Strategy selection logic based on query characteristics
        if analysis["boolean"] and len(analysis) < 3:
            # Boolean operators and few terms → BooleanAdapter (fastest)
            return "boolean"
        
        elif analysis["semantic"] and analysis["entity_count"] < 2:
            # Semantic but simple → DenseVectorAdapter
            return "dense_vector"
        
        elif analysis["type"] == "complex":
            # Complex query → Use hybrid fusion for coverage
            return "hybrid_fusion"
        
        elif analysis["type"] == "simple" and "temporal" in analysis:
            # Simple temporal → Cascading (filter then rank)
            return "cascading"
        
        else:
            # Default: BM25 (reliable, fast)
            return "bm25"
    
    def record_performance(
        self,
        query: str,
        strategy: str,
        metrics: dict,
    ):
        """Record performance for adaptive learning."""
        self.query_history.append({
            "query": query,
            "strategy": strategy,
            "metrics": metrics,
            "timestamp": time.time(),
        })
        
        # Update strategy performance tracking
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        self.strategy_performance[strategy].append(metrics)
    
    def learn_optimal_strategies(self):
        """Learn which strategies work best for which query types."""
        # Analyze patterns in query_history
        # Update selection logic based on actual performance
        pass
```

**Tests:**
- Query analysis correctness
- Strategy selection appropriateness
- Performance tracking
- Adaptive learning

**Metrics:**
- Strategy selection accuracy
- Query coverage (all types handled)
- Performance improvement over time

---

### T5.5: Personalized Memory Ranking (8 hours)

**Goal:** Rank retrieved memories based on user/agent preferences and history.

#### Implementation:

**PersonalizedMemoryRanker** (new class)
```python
class PersonalizedMemoryRanker:
    def __init__(self):
        """Initialize personalization profiles."""
        self.user_preferences = {}
        self.interaction_history = []
    
    def create_user_profile(self, user_id: str) -> dict:
        """Create or load user preference profile."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "topic_weights": {},
                "source_weights": {},
                "style_preferences": {},
                "recency_decay": 0.1,
                "feedback_history": [],
            }
        return self.user_preferences[user_id]
    
    def rank_memories(
        self,
        query: str,
        memories: list[Memory],
        user_id: str = None,
    ) -> list[Memory]:
        """Rank memories by user preferences."""
        if user_id is None:
            # No personalization, return as-is
            return memories
        
        profile = self.create_user_profile(user_id)
        
        # Compute personalization scores
        scores = {}
        for mem in memories:
            # 1. Topic affinity (does user prefer this topic?)
            topic_score = self._topic_affinity(mem, profile)
            
            # 2. Source preference (does user prefer this source?)
            source_score = self._source_preference(mem, profile)
            
            # 3. Style match (does memory match user's style?)
            style_score = self._style_match(mem, profile)
            
            # 4. Diversity (how different from other top results?)
            diversity_score = self._diversity_bonus(mem, scores)
            
            # Weighted combination
            scores[mem.id] = (
                0.40 * topic_score +
                0.25 * source_score +
                0.20 * style_score +
                0.15 * diversity_score
            )
        
        # Rerank by personalized scores
        ranked = sorted(
            memories,
            key=lambda m: scores[m.id],
            reverse=True
        )
        
        return ranked
    
    def record_feedback(
        self,
        user_id: str,
        memory_id: str,
        feedback: str,  # "liked", "disliked", "relevant", "irrelevant"
    ):
        """Update profile based on user feedback."""
        profile = self.create_user_profile(user_id)
        profile["feedback_history"].append({
            "memory_id": memory_id,
            "feedback": feedback,
            "timestamp": time.time(),
        })
        
        # Update topic/source weights based on feedback
        self._update_weights(profile, memory_id, feedback)
    
    def _topic_affinity(self, memory: Memory, profile: dict) -> float:
        """Score based on topic preferences."""
        topics = self._extract_topics(memory.content)
        weights = profile.get("topic_weights", {})
        return sum(weights.get(t, 0.5) for t in topics) / max(len(topics), 1)
    
    def _source_preference(self, memory: Memory, profile: dict) -> float:
        """Score based on source preferences."""
        source = memory.metadata.get("source", "unknown")
        weights = profile.get("source_weights", {})
        return weights.get(source, 0.5)
    
    def _style_match(self, memory: Memory, profile: dict) -> float:
        """Score based on style preferences."""
        style = self._analyze_style(memory.content)
        preferences = profile.get("style_preferences", {})
        match = sum(1 for k, v in preferences.items() if v == style.get(k))
        return match / max(len(preferences), 1)
    
    def _diversity_bonus(self, memory: Memory, scores: dict) -> float:
        """Bonus if memory is diverse from already-selected."""
        if not scores:
            return 1.0
        # Penalize if too similar to high-scoring memories
        return 0.5 + 0.5 * (1.0 - max(
            self._similarity(memory, m) for m in scores.keys()
        ))
```

**Tests:**
- Profile creation and updates
- Personalization scoring
- Feedback integration
- Diversity calculation
- Ranking consistency

**Metrics:**
- Personalization effectiveness (user engagement)
- Feedback impact (does ranking improve?)
- Diversity in results
- Profile convergence

---

### T5.6: End-to-End Integration & Testing (10 hours)

**Goal:** Integrate all Phase 5 components with Phase 4 retrieval framework.

#### Implementation:

**AdvancedMemorySystem** (orchestrator)
```python
class AdvancedMemorySystem:
    def __init__(self, retrieval_orchestrator):
        """Initialize complete memory system."""
        self.retrieval = retrieval_orchestrator
        self.clustering = SemanticClusterManager()
        self.attention = AttentionWeighter()
        self.consolidation = MemoryConsolidationEngine()
        self.strategy_selector = AdaptiveStrategySelector(retrieval_orchestrator.registry)
        self.personalizer = PersonalizedMemoryRanker()
    
    def process_query(
        self,
        query: str,
        user_id: str = None,
        constraints: dict = None,
    ) -> dict:
        """Full query processing pipeline."""
        # 1. Analyze query and select strategy
        strategy = self.strategy_selector.select_strategy(query, constraints)
        
        # 2. Find relevant cluster
        cluster_id = self.clustering.find_similar_cluster(query)
        cluster_memories = self.clustering.clusters.get(cluster_id, [])
        
        # 3. Retrieve using selected strategy
        adapter = self.retrieval.registry.get(strategy)
        candidates = adapter.search(query, top_k=20)
        
        # 4. Apply attention weighting
        weights = self.attention.adaptive_weights(self.strategy_selector.analyze_query(query)["type"])
        attention_scores = self.attention.compute_attention_scores(query, candidates, weights)
        
        # 5. Apply personalization if user provided
        if user_id:
            memories = self.personalizer.rank_memories(query, candidates, user_id)
        else:
            memories = sorted(candidates, key=lambda m: attention_scores.get(m.id, 0), reverse=True)
        
        # 6. Update consolidation state
        self.consolidation.consolidate_working_to_episodic()
        self.consolidation.consolidate_episodic_to_semantic()
        self.consolidation.decay_working_memory()
        
        # 7. Record metrics
        self.strategy_selector.record_performance(
            query,
            strategy,
            {"top_result_rank": 1, "num_results": len(memories)}
        )
        
        return {
            "results": memories[:10],
            "strategy": strategy,
            "cluster": cluster_id,
            "memory_stats": self.consolidation.get_memory_stats(),
        }
```

**Integration Tests:**
- Full query pipeline correctness
- Component interaction
- End-to-end performance
- Multi-user scenarios
- Failure recovery

---

## Deliverables

### Code (Phase 5)
- `benchmark/memory/advanced_strategies/`
  - `__init__.py`
  - `clustering.py` - SemanticClusterManager
  - `attention.py` - AttentionWeighter
  - `consolidation.py` - MemoryConsolidationEngine
  - `strategy_selector.py` - AdaptiveStrategySelector
  - `personalizer.py` - PersonalizedMemoryRanker
  - `advanced_system.py` - AdvancedMemorySystem orchestrator

### Tests
- `tests/test_semantic_clustering.py` (15 tests)
- `tests/test_attention_weighting.py` (12 tests)
- `tests/test_consolidation.py` (18 tests)
- `tests/test_strategy_selector.py` (10 tests)
- `tests/test_personalizer.py` (14 tests)
- `tests/test_advanced_system_integration.py` (20 tests)

### Documentation
- `PHASE5_ADVANCED_MEMORY_STRATEGIES_COMPLETE.md` - Implementation guide
- `PHASE5_USAGE_GUIDE.md` - How to use each component
- `PHASE5_ARCHITECTURE.md` - System design deep dive

### Benchmarks
- Sample execution results on all 14 Phase 3 datasets
- Performance comparisons: Phase 4 only vs Phase 5 enhanced
- Memory system metrics: consolidation rate, efficiency, coverage

---

## Success Criteria

### Code Quality
- ✓ All 89 tests passing (100%)
- ✓ Type hints on all functions
- ✓ Comprehensive docstrings
- ✓ Error handling for all edge cases
- ✓ No circular dependencies

### Performance
- ✓ Query processing < 100ms with personalization
- ✓ Consolidation doesn't impact retrieval
- ✓ Memory overhead < 20% of total
- ✓ Clustering stability (silhouette > 0.6)

### Integration
- ✓ Seamless Phase 4 integration
- ✓ Can use any Phase 4 retriever
- ✓ Backward compatible
- ✓ Works with all 14 datasets

---

## Timeline

| Week | Task | Hours | Deliverables |
|------|------|-------|--------------|
| 1 | T5.1 Clustering | 8 | SemanticClusterManager + tests |
| 2 | T5.2 Attention | 8 | AttentionWeighter + tests |
| 2 | T5.3 Consolidation | 10 | MemoryConsolidationEngine + tests |
| 3 | T5.4 Strategy Selection | 6 | AdaptiveStrategySelector + tests |
| 3 | T5.5 Personalization | 8 | PersonalizedMemoryRanker + tests |
| 4 | T5.6 Integration | 10 | AdvancedMemorySystem + tests |
| 5 | Documentation & Polish | 10 | Guides, examples, refinements |

**Total: 40 hours over 5 weeks**

---

## Next Steps After Phase 5

1. **Evaluation Pipeline** - Systematic testing across all 14 datasets
2. **Performance Optimization** - Profile and optimize bottlenecks
3. **Production Deployment** - Package for distribution
4. **User Studies** - Validate effectiveness with real agents
5. **Phase 6** - Advanced topics (continual learning, domain adaptation)

---

## References

- **Semantic Clustering:** K-means, DBSCAN, hierarchical clustering
- **Attention Mechanisms:** Multi-headed attention, soft attention
- **Memory Consolidation:** Sleep-dependent consolidation models (neuroscience inspiration)
- **Query Understanding:** NLU, query classification, entity recognition
- **Personalization:** Collaborative filtering, content-based filtering, hybrid approaches
- **Ranking:** Learning-to-rank, pointwise/pairwise/listwise objectives

---

## Conclusion

Phase 5 transforms Phase 4's retrieval framework into a **sophisticated agentic memory system** capable of:
- Intelligent memory organization (clustering)
- Context-aware retrieval (attention + strategy selection)
- Long-term memory management (consolidation)
- User-specific optimization (personalization)

This creates a foundation for truly adaptive, learning-capable agent memory systems.
