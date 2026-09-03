#!/usr/bin/env python3
"""Download and convert the extended benchmark datasets to gold format.

Covers the seven datasets not handled by prepare_datasets.py:
FEVER, MS MARCO, MultiWOZ 2.2, NarrativeQA, Natural Questions,
WebQuestions, and Wizard of Wikipedia.

Usage:
    python scripts/prepare_extended_datasets.py             # all seven
    python scripts/prepare_extended_datasets.py fever nq    # a subset

Set HF_TOKEN in .env (or the environment) for the HuggingFace sources.

Sources (dev/validation splits — small, publicly downloadable):
    fever        fever.ai shared task dev (19,998 claims, ~5 MB)
    msmarco      HF microsoft/ms_marco v1.1 validation parquet (~21 MB)
    multiwoz     GitHub budzianowski/multiwoz MultiWOZ_2.2 dev (512 dialogues)
    narrativeqa  GitHub deepmind/narrativeqa qaps.csv + summaries.csv (test split)
    nq           HF LLukas22/nq-simplified test.json (first 2,000 questions)
    webquestions HF stanfordnlp/web_questions test parquet (2,032 questions)
    wizard       HF chujiezheng/wizard_of_wikipedia valid_collected.json

Note: the official Natural Questions GCS bucket and the CMU HotpotQA server
both reject anonymous downloads at the time of writing, hence the HF mirrors.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data" / "input"


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  ✓ already present: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("HF_TOKEN", "").strip()
    if "huggingface.co" in url and token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"  downloading {url.split('/')[-1]} ...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        while block := resp.read(1 << 16):
            f.write(block)
    return dest


def _parquet_rows(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pyarrow required for parquet sources: pip install pyarrow")
    return pq.read_table(path).to_pylist()


def _convert(adapter_module: str, adapter_class: str, src: Path, out: Path) -> None:
    adapter = getattr(importlib.import_module(adapter_module), adapter_class)()
    dataset = adapter.load(src)
    out.write_text(dataset.model_dump_json(indent=2))
    n_mem = sum(len(e.memory_events) for e in dataset.events)
    print(f"  ✓ {out.name}: {len(dataset.queries)} queries, {n_mem} memories")


def prepare_fever() -> None:
    src = _fetch(
        "https://fever.ai/download/fever/shared_task_dev.jsonl",
        DATA_DIR / "fever" / "shared_task_dev.jsonl",
    )
    _convert("benchmark.gold.adapters.fever_adapter", "FEVERAdapter", src, DATA_DIR / "fever_gold.json")


def prepare_msmarco() -> None:
    pq_path = _fetch(
        "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/validation-00000-of-00001.parquet",
        DATA_DIR / "msmarco" / "msmarco_validation.parquet",
    )
    src = DATA_DIR / "msmarco" / "msmarco_dev.jsonl"
    if not src.exists():
        with open(src, "w") as f:
            for r in _parquet_rows(pq_path):
                passages = [{"passage_text": t} for t in r["passages"]["passage_text"]]
                f.write(json.dumps({"query": r["query"], "passages": passages}) + "\n")
    _convert("benchmark.gold.adapters.msmarco_adapter", "MSMarcoAdapter", src, DATA_DIR / "msmarco_gold.json")


def prepare_multiwoz() -> None:
    raw = _fetch(
        "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/dev/dialogues_001.json",
        DATA_DIR / "multiwoz" / "dialogues_001.json",
    )
    src = DATA_DIR / "multiwoz" / "multiwoz22_dev.json"
    if not src.exists():
        with open(raw) as f:
            dialogues = json.load(f)
        as_dict = {
            d["dialogue_id"]: {
                "turns": [{"speaker": t["speaker"], "utterance": t["utterance"]} for t in d["turns"]]
            }
            for d in dialogues
        }
        with open(src, "w") as f:
            json.dump(as_dict, f)
    _convert("benchmark.gold.adapters.multiwoz_adapter", "MultiWOZAdapter", src, DATA_DIR / "multiwoz_gold.json")


def prepare_narrativeqa() -> None:
    qaps = _fetch(
        "https://raw.githubusercontent.com/deepmind/narrativeqa/master/qaps.csv",
        DATA_DIR / "narrativeqa" / "qaps.csv",
    )
    summaries = _fetch(
        "https://raw.githubusercontent.com/deepmind/narrativeqa/master/third_party/wikipedia/summaries.csv",
        DATA_DIR / "narrativeqa" / "summaries.csv",
    )
    src = DATA_DIR / "narrativeqa" / "narrativeqa_test.json"
    if not src.exists():
        stories = {}
        with open(summaries) as f:
            for r in csv.DictReader(f):
                if r["set"] == "test":
                    stories[r["document_id"]] = r["summary"]
        questions = defaultdict(list)
        with open(qaps) as f:
            for r in csv.DictReader(f):
                if r["set"] == "test" and r["document_id"] in stories:
                    questions[r["document_id"]].append(r["question"])
        data = [{"story": stories[d], "questions": qs} for d, qs in questions.items()]
        with open(src, "w") as f:
            json.dump(data, f)
    _convert(
        "benchmark.gold.adapters.narrativeqa_adapter", "NarrativeQAAdapter", src, DATA_DIR / "narrativeqa_gold.json"
    )


def prepare_nq(sample_size: int = 2000) -> None:
    raw = _fetch(
        "https://huggingface.co/datasets/LLukas22/nq-simplified/resolve/main/test.json",
        DATA_DIR / "naturalquestions" / "nq_simplified_test.jsonl",
    )
    src = DATA_DIR / "naturalquestions" / "nq_dev_sample.jsonl"
    if not src.exists():
        with open(raw) as f, open(src, "w") as out:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                r = json.loads(line)
                out.write(json.dumps({"question": r["question"], "document_text": r["context"]}) + "\n")
    _convert(
        "benchmark.gold.adapters.naturalquestions_adapter",
        "NaturalQuestionsAdapter",
        src,
        DATA_DIR / "naturalquestions_gold.json",
    )


def prepare_webquestions() -> None:
    pq_path = _fetch(
        "https://huggingface.co/datasets/stanfordnlp/web_questions/resolve/main/data/test-00000-of-00001.parquet",
        DATA_DIR / "webquestions" / "webquestions_test.parquet",
    )
    src = DATA_DIR / "webquestions" / "webquestions_test.jsonl"
    if not src.exists():
        with open(src, "w") as f:
            for r in _parquet_rows(pq_path):
                f.write(json.dumps({"question": r["question"], "answers": list(r["answers"])}) + "\n")
    _convert(
        "benchmark.gold.adapters.webquestions_adapter", "WebQuestionsAdapter", src, DATA_DIR / "webquestions_gold.json"
    )


def prepare_wizard() -> None:
    raw = _fetch(
        "https://huggingface.co/datasets/chujiezheng/wizard_of_wikipedia/resolve/main/valid_collected.json",
        DATA_DIR / "wizard" / "valid_collected.json",
    )
    src = DATA_DIR / "wizard" / "wizard_valid.jsonl"
    if not src.exists():
        no_passage = "no_passages_used __knowledge__ no_passages_used"
        with open(raw) as f:
            records = json.load(f)
        with open(src, "w") as out:
            for r in records:
                knowledge, seen = [], set()
                for i, candidates in enumerate(r["knowledge"]):
                    label = r["labels"][i]
                    if isinstance(label, int) and 0 <= label < len(candidates):
                        gold = candidates[label]
                        if gold and gold != no_passage and gold not in seen:
                            seen.add(gold)
                            knowledge.append(gold.replace(" __knowledge__ ", ": "))
                out.write(json.dumps({"knowledge": knowledge, "history": [p for p in r["post"] if p]}) + "\n")
    _convert("benchmark.gold.adapters.wizard_adapter", "WizardAdapter", src, DATA_DIR / "wizard_gold.json")


PREPARERS = {
    "fever": prepare_fever,
    "msmarco": prepare_msmarco,
    "multiwoz": prepare_multiwoz,
    "narrativeqa": prepare_narrativeqa,
    "nq": prepare_nq,
    "webquestions": prepare_webquestions,
    "wizard": prepare_wizard,
}


def load_env() -> None:
    """Load .env for HF_TOKEN without requiring python-dotenv."""
    env_file = project_root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env()
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and not token.startswith("hf_your"):
        print("  HF_TOKEN found — authenticated HuggingFace downloads enabled.")
    else:
        print("  HF_TOKEN not set — HuggingFace sources (msmarco, nq, webquestions, wizard)")
        print("  will be attempted anonymously and may fail if rate-limited or gated.")
        print("  Fix: add HF_TOKEN=hf_... to .env (https://huggingface.co/settings/tokens)")

    names = sys.argv[1:] or list(PREPARERS)
    for name in names:
        if name not in PREPARERS:
            print(f"unknown dataset: {name} (choose from {', '.join(PREPARERS)})")
            continue
        print(f"\n=== {name} ===")
        try:
            PREPARERS[name]()
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")


if __name__ == "__main__":
    main()
