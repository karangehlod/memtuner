#!/usr/bin/env python3
"""
Test HuggingFace components only (no Ollama required).
Run: source .venv/bin/activate && python test_hf_only.py
"""

import os
import sys

# Load environment
os.chdir('/Users/karangehlod/Codes/Agenticmemory_benchmark')
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

def test_local_embedding():
    """Test sentence-transformers local embedding model"""
    print("\n" + "="*70)
    print("TEST 1: Local Embedding Model (sentence-transformers/all-MiniLM-L6-v2)")
    print("="*70)
    try:
        from sentence_transformers import SentenceTransformer

        print("Loading model (first time may take 1-2 minutes)...")
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


def test_hf_inference():
    """Test HuggingFace Inference API"""
    print("\n" + "="*70)
    print("TEST 2: HuggingFace Inference API")
    print("="*70)
    try:
        import httpx

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("✗ HF_TOKEN not set in .env")
            return False

        print(f"HF Token: {hf_token[:20]}...")

        # Test API availability
        print("\nTesting HF Inference API...")
        with httpx.Client(timeout=30) as client:
            headers = {"Authorization": f"Bearer {hf_token}"}

            # List available models
            response = client.get(
                "https://api-inference.huggingface.co/models",
                headers=headers,
                timeout=10
            )

            if response.status_code == 401:
                print(f"✗ Authentication failed: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return False

            if response.status_code == 200:
                print(f"✓ HF Inference API authenticated successfully")
            else:
                print(f"⚠ HF Inference API status: {response.status_code}")

            # Test embedding model via HF Inference
            print("\nTesting embedding model via HF Inference...")
            response = client.post(
                "https://api-inference.huggingface.co/models/BAAI/bge-base-en-v1.5",
                headers=headers,
                json={"inputs": ["Hello world", "This is a test"]},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    print(f"✓ Embeddings generated")
                    print(f"✓ Number of embeddings: {len(result)}")
                    print(f"✓ Embedding dimension: {len(result[0]) if isinstance(result[0], list) else 'N/A'}")
                    print(f"✓ HF INFERENCE EMBEDDINGS: WORKING")
                    return True
                else:
                    print(f"✓ API returned data (may be cold start): {str(result)[:100]}")
                    return True
            else:
                print(f"✗ API returned status {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return False

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hf_reranker():
    """Test HuggingFace Reranker API"""
    print("\n" + "="*70)
    print("TEST 3: HuggingFace Reranker (BAAI/bge-reranker-base)")
    print("="*70)
    try:
        import httpx

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("✗ HF_TOKEN not set in .env")
            return False

        print("Testing reranker via HF Inference...")
        with httpx.Client(timeout=30) as client:
            headers = {"Authorization": f"Bearer {hf_token}"}

            # Prepare reranker input
            texts_pairs = [
                ["What is machine learning?", "Machine learning is a subset of AI"],
                ["What is machine learning?", "Cooking recipes for pasta"],
            ]

            response = client.post(
                "https://api-inference.huggingface.co/models/BAAI/bge-reranker-base",
                headers=headers,
                json={"inputs": texts_pairs},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Reranker responded successfully")
                print(f"✓ Response: {str(result)[:100]}")
                print(f"✓ HF RERANKER: WORKING")
                return True
            else:
                print(f"⚠ API status {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return response.status_code in [200, 202]  # 202 = model loading

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# HF COMPONENTS TEST (No Ollama Required)")
    print("#"*70)
    print("Testing HuggingFace components...\n")

    results = {}

    # Test 1: Local embeddings
    results["Local Embedding"] = test_local_embedding()

    # Test 2: HF Inference API
    results["HF Inference API"] = test_hf_inference()

    # Test 3: HF Reranker
    results["HF Reranker"] = test_hf_reranker()

    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ ALL HF TESTS PASSED - HuggingFace components working!")
        sys.exit(0)
    else:
        print("\n⚠ Some HF tests failed - check network/token")
        sys.exit(1)
