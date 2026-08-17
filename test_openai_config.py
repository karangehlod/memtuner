#!/usr/bin/env python3
"""
Test OpenAI configuration and model support.
Tests the new provider-agnostic embedding and LLM infrastructure.
"""

import os
import sys

os.chdir('/Users/karangehlod/Codes/Agenticmemory_benchmark')
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

print("="*70)
print("OpenAI Configuration Test")
print("="*70)

# Test 1: Provider Service
print("\n1. Testing Provider Service...")
try:
    from benchmark.services.provider_service import ProviderService

    # Get embedding provider
    embedding_provider = ProviderService.get_embedding_provider()
    print(f"   ✓ Embedding provider: {embedding_provider}")

    # Get LLM provider
    llm_provider = ProviderService.get_llm_provider()
    print(f"   ✓ LLM provider: {llm_provider}")

    # Get judge provider
    judge_provider = ProviderService.get_judge_provider()
    print(f"   ✓ Judge provider: {judge_provider}")

except Exception as e:
    print(f"   ✗ Provider Service failed: {e}")
    sys.exit(1)

# Test 2: Universal Embeddings Strategy
print("\n2. Testing Universal Embeddings Strategy...")
try:
    from benchmark.memory.strategies.universal_embeddings_strategy import (
        UniversalEmbeddingsStrategy
    )
    from benchmark.services.provider_service import ProviderType, ProviderConfig

    # Test with current provider
    strategy = UniversalEmbeddingsStrategy(
        provider_config=embedding_provider,
        timeout=120.0
    )
    print(f"   ✓ Strategy created: {strategy}")

    # Try embedding sample texts
    test_texts = [
        "What is machine learning?",
        "Machine learning is a subset of AI",
        "Neural networks process data"
    ]

    print(f"   Testing embedding of {len(test_texts)} sample texts...")
    embeddings = strategy.embed(test_texts)
    print(f"   ✓ Got embeddings: shape={embeddings.shape}")

except Exception as e:
    print(f"   ✗ Universal Embeddings Strategy failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model Tester
print("\n3. Testing Model Tester...")
try:
    from benchmark.services.model_tester import ModelTester

    print("   Getting available embedding models...")
    all_models = ModelTester.get_all_embedding_models()
    print(f"   ✓ Found {len(all_models)} embedding models:")
    for model in all_models:
        print(f"      - {model.provider_type.value}: {model.model_name}")

    print("\n   Testing Ollama embedding models...")
    ollama_models = ModelTester.get_ollama_embedding_models()
    test_texts = ["test query 1", "test query 2"]

    for model in ollama_models[:2]:  # Test first 2 only
        print(f"   Testing {model.provider_type.value}/{model.model_name}...")
        result = ModelTester.test_embedding_model(model, test_texts)
        if result.success:
            print(f"      ✓ Success")
        else:
            print(f"      ✗ Failed: {result.error}")

except Exception as e:
    print(f"   ⚠ Model Tester warning: {e}")
    # Don't exit on error here - tests may fail due to network

# Test 4: Configuration Summary
print("\n4. Configuration Summary")
print("="*70)
print(f"Embedding Provider: {embedding_provider.provider_type.value}")
print(f"  - Base URL: {embedding_provider.base_url}")
print(f"  - Model: {embedding_provider.model_name}")
print(f"  - Timeout: {embedding_provider.timeout}s")

print(f"\nLLM Provider: {llm_provider.provider_type.value}")
print(f"  - Base URL: {llm_provider.base_url}")
print(f"  - Model: {llm_provider.model_name}")
print(f"  - Timeout: {llm_provider.timeout}s")

print(f"\nJudge Provider: {judge_provider.provider_type.value}")
print(f"  - Base URL: {judge_provider.base_url}")
print(f"  - Model: {judge_provider.model_name}")
print(f"  - Timeout: {judge_provider.timeout}s")

print("\n" + "="*70)
print("✓ OpenAI Configuration Test Complete")
print("="*70)
print("\nTo switch providers, set environment variables:")
print("  BENCHMARK_EMBEDDING_PROVIDER=openai|ollama|huggingface")
print("  BENCHMARK_LLM_PROVIDER=openai|ollama|anthropic")
print("  BENCHMARK_JUDGE_PROVIDER=openai|ollama|anthropic")
print("\nAnd corresponding API keys:")
print("  OPENAI_API_KEY=sk-...")
print("  ANTHROPIC_API_KEY=sk-ant-...")
print("  HF_TOKEN=hf_...")
