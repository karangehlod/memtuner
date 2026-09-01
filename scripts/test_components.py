#!/usr/bin/env python3
"""
Standalone test script to verify individual benchmark components work correctly.
Run in venv: python test_components.py

This tests:
1. Local embedding model (sentence-transformers)
2. Ollama embedding models
3. Ollama reranker (LLM-based)

Environment variables:
- OLLAMA_BASE_URL: Ollama API base URL (default: http://localhost:11434/v1)
  Example: OLLAMA_BASE_URL=http://192.168.1.100:11434/v1 python test_components.py
"""

import sys
import os

# Get Ollama URL from environment or use default
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
print(f"Using Ollama URL: {OLLAMA_BASE_URL}")

def test_local_embedding():
    """Test sentence-transformers local embedding model"""
    print("\n" + "="*70)
    print("TEST 1: Local Embedding Model (sentence-transformers/all-MiniLM-L6-v2)")
    print("="*70)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = ["Hello world", "This is a test", "Embedding models work"]
        embeddings = model.encode(texts)

        print(f"✓ Model loaded successfully")
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"✓ Embedding dimension: {embeddings[0].shape}")
        print(f"✓ LOCAL EMBEDDING MODEL: WORKING")
        return True
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama_embeddings():
    """Test Ollama embedding models"""
    print("\n" + "="*70)
    print("TEST 2: Ollama Embedding Models")
    print("="*70)
    try:
        import httpx
        from benchmark.memory.strategies.ollama_embeddings_strategy import OllamaEmbeddingsStrategy

        # Check Ollama is running - use /api/tags endpoint (more reliable)
        print(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")

        # Build base URL for /api/tags (strip /v1 suffix if present)
        api_base_url = OLLAMA_BASE_URL.replace("/v1", "")

        with httpx.Client(base_url=api_base_url, timeout=5) as client:
            response = client.get("/api/tags")
            models_data = response.json()
            models_list = models_data.get("models", [])

            # Extract model names
            model_names = []
            for m in models_list:
                if isinstance(m, dict):
                    model_names.append(m.get('name', m.get('id', str(m))))
                else:
                    model_names.append(str(m))

            print(f"✓ Ollama is running")
            print(f"✓ Available {len(model_names)} models")
            for i, name in enumerate(model_names[:5]):
                print(f"  {i+1}. {name}")

            # Check for embedding models - look for common embedding model names
            embedding_models = [m for m in model_names if any(x in m.lower() for x in ['embed', 'nomic', 'bge', 'gemma', 'qwen'])]
            if embedding_models:
                print(f"✓ Found {len(embedding_models)} embedding model(s): {embedding_models}")
                selected_model = embedding_models[0]  # Use first available embedding model
            else:
                print(f"⚠ No embedding models found in available models: {model_names}")
                print(f"  Trying with 'nomic-embed-text' anyway...")
                selected_model = "nomic-embed-text"

        # Try creating and using the strategy
        print(f"\nTesting OllamaEmbeddingsStrategy with {selected_model}...")
        strategy = OllamaEmbeddingsStrategy(
            model_name=selected_model,
            base_url=OLLAMA_BASE_URL,
            api_key="not-needed",
            timeout=30.0  # Add required timeout parameter
        )
        print(f"✓ OllamaEmbeddingsStrategy created")

        # Try embedding
        test_texts = ["test query", "another query"]
        embeddings = strategy.embed(test_texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"✓ Embedding dimension: {embeddings[0].shape if hasattr(embeddings[0], 'shape') else len(embeddings[0])}")
        print(f"✓ OLLAMA EMBEDDING MODELS: WORKING")
        return True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama_reranker():
    """Test Ollama LLM for reranking"""
    print("\n" + "="*70)
    print("TEST 3: Ollama Reranker (via LLM)")
    print("="*70)
    try:
        import httpx

        print(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")

        # Build base URL for /api/tags (strip /v1 suffix if present)
        api_base_url = OLLAMA_BASE_URL.replace("/v1", "")

        # Check LLM model is available
        with httpx.Client(base_url=api_base_url, timeout=5) as client:
            response = client.get("/api/tags")
            models_data = response.json()
            models_list = models_data.get("models", [])

            model_names = []
            for m in models_list:
                if isinstance(m, dict):
                    model_names.append(m.get('name', m.get('id', str(m))))
                else:
                    model_names.append(str(m))

            print(f"✓ Ollama is running")
            print(f"✓ Available {len(model_names)} models")

            # Check for nemotron or other LLM models
            llm_models = [m for m in model_names if any(x in m.lower() for x in ['nemotron', 'gemma', 'llama'])]
            if llm_models:
                print(f"✓ Found {len(llm_models)} LLM model(s): {llm_models}")
                selected_llm = llm_models[0]
            else:
                print(f"⚠ No known LLM models found")
                print(f"  Available: {model_names}")
                selected_llm = "nemotron-3-nano:4b"  # Try default

        # Try an LLM call for reranking
        print(f"\nTesting LLM reranking capability with {selected_llm}...")
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=30) as client:
            response = client.post(
                "/chat/completions",
                json={
                    "model": selected_llm,
                    "messages": [
                        {"role": "user", "content": "Rank these by relevance to 'machine learning': A. Python programming, B. Cooking recipes, C. Neural networks"}
                    ],
                    "max_tokens": 100,
                },
                timeout=30
            )

            if response.status_code != 200:
                print(f"✗ HTTP Error: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

            result = response.json()
            msg = result.get('choices', [{}])[0].get('message', {}).get('content', '')

            print(f"✓ Ollama LLM responded successfully")
            print(f"✓ Model response preview: {msg[:100]}")
            print(f"✓ OLLAMA RERANKER (LLM): WORKING")
            return True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# BENCHMARK COMPONENT TESTS")
    print("#"*70)
    print("Testing individual embedding and reranker components...\n")

    results = {}

    # Test 1: Local embeddings
    results["Local Embedding"] = test_local_embedding()

    # Test 2: Ollama embeddings
    results["Ollama Embeddings"] = test_ollama_embeddings()

    # Test 3: Ollama reranker
    results["Ollama Reranker"] = test_ollama_reranker()

    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Components are working correctly")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED - Fix issues above before running benchmark")
        sys.exit(1)
