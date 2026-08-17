"""LLM-as-Judge module for end-to-end answer quality evaluation.

Uses any OpenAI-compatible endpoint (Ollama, vLLM, HuggingFace TGI, OpenAI)
to evaluate whether retrieved memories produce correct answers.

Environment variables:
    BENCHMARK_LLM_BASE_URL: API endpoint (default: http://localhost:11434/v1)
    BENCHMARK_LLM_API_KEY: API key (default: "not-needed" for local models)
    BENCHMARK_LLM_MODEL: Model name (default: "llama3.1:8b")
    BENCHMARK_JUDGE_MODEL: Judge model (default: same as LLM_MODEL)
"""
