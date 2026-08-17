from benchmark.tokenizer.bpe import SimpleBPETokenizer
from benchmark.cost.token_cost import TokenCostCalculator
from benchmark.models.answer import TokenUsage


def test_simple_bpe_counts_and_token_cost():
    tokenizer = SimpleBPETokenizer()
    text = "Alice prefers Postgres over Pinecone."
    count = tokenizer.count_tokens(text)
    assert count > 0
    enc = tokenizer.encode(text)
    assert isinstance(enc, list)

    tcc = TokenCostCalculator(tokenizer=tokenizer)
    # compute via tokenizer (no TokenUsage provided)
    entry = tcc.compute_cost(None, "gpt-4o", prompt_text=text, completion_text="OK")
    assert entry.amount_usd >= 0.0

    # compute via explicit TokenUsage
    usage = TokenUsage(prompt=count, completion=1)
    entry2 = tcc.compute_cost(usage, "gpt-4o")
    assert abs(entry2.amount_usd - entry.amount_usd) < 1e-6
