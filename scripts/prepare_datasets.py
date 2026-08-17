#!/usr/bin/env python3
"""Download and convert all supported benchmark datasets to gold format.

Usage:
    python scripts/prepare_datasets.py                  # show status only
    python scripts/prepare_datasets.py --download       # download missing files
    python scripts/prepare_datasets.py --convert        # convert downloaded files
    python scripts/prepare_datasets.py --download --convert  # do both

Datasets:
    locomo       Already works natively — no conversion needed.
    longmemeval  Needs download + convert.
    squad        Download from Stanford NLP (public, no auth).
    coqa         Download from Stanford NLP (public, no auth).
    personachat  Download from HuggingFace (public, no auth).
    hotpotqa     Download from HuggingFace (public, no auth).
    synthetic    No download — generated on demand, zero setup.

Not currently downloadable automatically (require HF auth or large size):
    fever        185k claims — requires manual download from ai.facebook.com
    msmarco      100k+ queries — use official MS MARCO tools
    multiwoz     10k dialogues — requires manual download
    narrativeqa  31k questions — requires manual download
    naturalquestions  320k questions — requires manual download
    wizard       18k dialogues — requires manual download
    webquestions 5k questions — requires manual download
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data" / "input"

# ── Download specs ──────────────────────────────────────────────────────────
# Each entry: (local_path, url, description)
DOWNLOADS: list[tuple[Path, str, str]] = [
    (
        DATA_DIR / "longmemeval" / "longmemeval_oracle.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
        "LongMemEval oracle (500 questions, ~30 MB)",
    ),
    (
        DATA_DIR / "squad" / "squad_dev-v2.0.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "SQuAD 2.0 dev set (~36 MB)",
    ),
    (
        DATA_DIR / "coqa" / "coqa-dev-v1.0.json",
        "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json",
        "CoQA dev set (~55 MB)",
    ),
]

# ── Conversion specs ────────────────────────────────────────────────────────
def _convert_longmemeval() -> Path:
    from benchmark.gold.longmemeval_adapter import convert_longmemeval_to_gold
    src = DATA_DIR / "longmemeval" / "longmemeval_oracle.json"
    out = DATA_DIR / "longmemeval_oracle_gold.json"
    convert_longmemeval_to_gold(src, out, "longmemeval_full")
    return out


def _convert_squad() -> Path:
    from benchmark.gold.adapters.squad_adapter import SQuADAdapter
    src = DATA_DIR / "squad" / "squad_dev-v2.0.json"
    out = DATA_DIR / "squad_gold.json"
    dataset = SQuADAdapter().load(src)
    out.write_text(dataset.model_dump_json(indent=2))
    return out


def _convert_coqa() -> Path:
    from benchmark.gold.adapters.coqa_adapter import CoQAAdapter
    src = DATA_DIR / "coqa" / "coqa-dev-v1.0.json"
    out = DATA_DIR / "coqa_gold.json"
    dataset = CoQAAdapter().load(src)
    out.write_text(dataset.model_dump_json(indent=2))
    return out


def _generate_synthetic() -> Path:
    from benchmark.gold.adapters.synthetic_adapter import SyntheticAdapter
    out = DATA_DIR / "synthetic_gold.json"
    dataset = SyntheticAdapter(query_count=200, user_count=5, day_range=50, seed=42).load()
    out.write_text(dataset.model_dump_json(indent=2))
    return out


CONVERSIONS: list[tuple[str, Path, callable, Path]] = [
    # (name, required_source, converter_fn, output_path)
    ("longmemeval", DATA_DIR / "longmemeval" / "longmemeval_oracle.json", _convert_longmemeval, DATA_DIR / "longmemeval_oracle_gold.json"),
    ("squad",       DATA_DIR / "squad" / "squad_dev-v2.0.json",          _convert_squad,        DATA_DIR / "squad_gold.json"),
    ("coqa",        DATA_DIR / "coqa" / "coqa-dev-v1.0.json",            _convert_coqa,         DATA_DIR / "coqa_gold.json"),
    ("synthetic",   None,                                                  _generate_synthetic,   DATA_DIR / "synthetic_gold.json"),
]

# ── Manual-download datasets (can't be auto-fetched) ───────────────────────
MANUAL_DOWNLOADS = [
    ("fever",           "https://fever.ai/dataset/fever.html",                           "train.jsonl → data/fever/"),
    ("msmarco",         "https://microsoft.github.io/msmarco/",                          "queries.jsonl + passages.jsonl → data/msmarco/"),
    ("multiwoz",        "https://github.com/budzianowski/multiwoz",                      "data.json → data/multiwoz/"),
    ("narrativeqa",     "https://github.com/deepmind/narrativeqa",                       "qaps.csv + summaries/ → data/narrativeqa/"),
    ("naturalquestions","https://ai.google.com/research/NaturalQuestions",               "nq-dev-*.jsonl.gz → data/naturalquestions/"),
    ("webquestions",    "https://github.com/brmson/dataset-factoid-webquestions",        "webquestions.examples.train.json → data/webquestions/"),
    ("wizard",          "https://parl.ai/projects/wizard_of_wikipedia/",                 "train.json → data/wizard/"),
    ("personachat",     "https://huggingface.co/datasets/bavard/personachat_truecased",  "train.json → data/personachat/"),
    ("hotpotqa",        "https://hotpotqa.github.io/",                                   "hotpot_dev_distractor_v1.json → data/hotpotqa/"),
]


def _download_file(url: str, dest: Path, description: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {description}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 64 * 1024
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r    {pct:3d}%  {downloaded // 1024 // 1024} MB / {total // 1024 // 1024} MB", end="", flush=True)
        print(f"\r    ✓ Saved to {dest}")
        return True
    except Exception as e:
        print(f"\r    ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def print_status() -> None:
    print("\n=== Dataset Status ===\n")

    print("AUTO (locomo — no conversion needed):")
    f = DATA_DIR / "locomo10.json"
    status = f"✓ {f.stat().st_size // 1024 // 1024} MB" if f.exists() else "✗ missing"
    print(f"  locomo10.json       {status}")
    if f.exists():
        print(f"    → Run: python study_runner.py --gold-dataset data/locomo10.json --mode full")

    print("\nAUTO-DOWNLOAD + CONVERT:")
    for name, src, _, out in CONVERSIONS:
        if src is None:
            src_status = "(generated)"
            out_status = f"✓ {out.stat().st_size // 1024} KB" if out.exists() else "✗ not yet generated"
        else:
            src_status = f"✓ {src.stat().st_size // 1024 // 1024} MB" if src.exists() else "✗ not downloaded"
            out_status = f"✓ {out.stat().st_size // 1024 // 1024} MB" if out.exists() else "✗ not converted"
        print(f"  {name:15s}  source={src_status:25s}  gold={out_status}")
        if out.exists():
            print(f"    → Run: python study_runner.py --gold-dataset {out.relative_to(project_root)} --mode full")

    print("\nMANUAL DOWNLOAD REQUIRED:")
    for name, url, instructions in MANUAL_DOWNLOADS:
        print(f"  {name:20s}  {url}")
        print(f"    Instructions: {instructions}")

    print()


def do_download() -> None:
    print("\n=== Downloading Datasets ===\n")
    for dest, url, desc in DOWNLOADS:
        if dest.exists():
            print(f"  ✓ Already present: {dest.name}")
            continue
        _download_file(url, dest, desc)


def do_convert() -> None:
    print("\n=== Converting to Gold Format ===\n")
    for name, src, converter_fn, out in CONVERSIONS:
        if out.exists():
            print(f"  ✓ Already converted: {out.name}")
            continue
        if src is not None and not src.exists():
            print(f"  ✗ {name}: source not found ({src}) — run --download first")
            continue
        print(f"  Converting {name}...")
        try:
            result = converter_fn()
            size = result.stat().st_size // 1024
            print(f"  ✓ {name}: {result.name} ({size} KB)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare benchmark datasets")
    parser.add_argument("--download", action="store_true", help="Download missing dataset files")
    parser.add_argument("--convert", action="store_true", help="Convert downloaded files to gold format")
    args = parser.parse_args()

    if args.download:
        do_download()
    if args.convert:
        do_convert()
    if not args.download and not args.convert:
        print_status()
        print("Run with --download to fetch missing files, --convert to convert them.")
        print("Both flags can be combined: --download --convert")


if __name__ == "__main__":
    main()
