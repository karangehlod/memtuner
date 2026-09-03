#!/usr/bin/env python3
"""Download and convert all supported benchmark datasets to gold format.

Usage:
    python scripts/prepare_datasets.py                  # show status only
    python scripts/prepare_datasets.py --download       # download missing files
    python scripts/prepare_datasets.py --convert        # convert downloaded files
    python scripts/prepare_datasets.py --download --convert  # do both

AUTO-DOWNLOADABLE datasets (set HF_TOKEN in .env for HuggingFace datasets):
    locomo       GitHub snap-research (~3 MB) — no auth needed.
    longmemeval  HuggingFace (~30 MB) — HF_TOKEN recommended.
    squad        Stanford NLP (~36 MB) — no auth needed.
    coqa         Stanford NLP (~55 MB) — no auth needed.
    personachat  HuggingFace (~20 MB) — set HF_TOKEN in .env; skipped if unavailable.
    hotpotqa     CMU (~54 MB) — no auth needed.
    synthetic    No download — generated on demand (200 queries, zero setup).

MANUAL download required (large size or special tools):
    fever        185k claims — https://fever.ai/dataset/fever.html
    msmarco      100k+ queries — https://microsoft.github.io/msmarco/
    multiwoz     10k dialogues — https://github.com/budzianowski/multiwoz
    narrativeqa  31k questions — https://github.com/deepmind/narrativeqa
    naturalquestions  320k questions — https://ai.google.com/research/NaturalQuestions
    wizard       18k dialogues — https://parl.ai/projects/wizard_of_wikipedia/
    webquestions 5k questions — https://github.com/brmson/dataset-factoid-webquestions
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data" / "input"

# ── Download specs ──────────────────────────────────────────────────────────
# Each entry: (local_path, url, description)
# HuggingFace URLs automatically include Authorization header when HF_TOKEN is set.
DOWNLOADS: list[tuple[Path, str, str]] = [
    (
        DATA_DIR / "locomo10.json",
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "LoCoMo (10 long conversations, ~3 MB, CC BY-NC 4.0 — Snap Research)",
    ),
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
    (
        DATA_DIR / "personachat" / "personachat_truecased_full_train.json",
        "https://huggingface.co/datasets/bavard/personachat_truecased/resolve/main/personachat_truecased_full_train.json",
        "PersonaChat truecased full train (~20 MB) — HF_TOKEN recommended",
    ),
    (
        DATA_DIR / "hotpotqa" / "hotpot_dev_distractor_v1.json",
        # Primary: CMU server (may be slow/down — retry if 504)
        # Fallback: download manually from https://hotpotqa.github.io/
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "HotpotQA distractor dev (~54 MB) — CMU server; retry if 504",
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


def _convert_personachat() -> Path:
    import json as _json

    from benchmark.gold.adapters.personachat_adapter import PersonaChatAdapter
    src = DATA_DIR / "personachat" / "personachat_truecased_full_train.json"
    out = DATA_DIR / "personachat_gold.json"
    # Use first 500 dialogues (~3,675 queries) — comparable to other benchmark splits.
    # Full train has 17,878 dialogues (88 MB gold) which is too large to commit.
    with open(src) as _f:
        full = _json.load(_f)
    subset_path = src.parent / "_subset_500.json"
    with open(subset_path, "w") as _f:
        _json.dump(full[:500], _f)
    dataset = PersonaChatAdapter().load(subset_path)
    subset_path.unlink()
    out.write_text(dataset.model_dump_json(indent=2))
    return out


def _convert_hotpotqa() -> Path:
    from benchmark.gold.adapters.hotpotqa_adapter import HotpotQAAdapter
    src = DATA_DIR / "hotpotqa" / "hotpot_dev_distractor_v1.json"
    out = DATA_DIR / "hotpotqa_gold.json"
    dataset = HotpotQAAdapter().load(src)
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
    ("personachat", DATA_DIR / "personachat" / "personachat_truecased_full_train.json",  _convert_personachat,  DATA_DIR / "personachat_gold.json"),
    ("hotpotqa",    DATA_DIR / "hotpotqa" / "hotpot_dev_distractor_v1.json", _convert_hotpotqa, DATA_DIR / "hotpotqa_gold.json"),
    ("synthetic",   None,                                                  _generate_synthetic,   DATA_DIR / "synthetic_gold.json"),
]

# ── Manual-download datasets (require auth, large size, or special tools) ──
MANUAL_DOWNLOADS = [
    ("fever",           "https://fever.ai/dataset/fever.html",                           "train.jsonl → data/fever/"),
    ("msmarco",         "https://microsoft.github.io/msmarco/",                          "queries.jsonl + passages.jsonl → data/msmarco/"),
    ("multiwoz",        "https://github.com/budzianowski/multiwoz",                      "data.json → data/multiwoz/"),
    ("narrativeqa",     "https://github.com/deepmind/narrativeqa",                       "qaps.csv + summaries/ → data/narrativeqa/"),
    ("naturalquestions","https://ai.google.com/research/NaturalQuestions",               "nq-dev-*.jsonl.gz → data/naturalquestions/"),
    ("webquestions",    "https://github.com/brmson/dataset-factoid-webquestions",        "webquestions.examples.train.json → data/webquestions/"),
    ("wizard",          "https://parl.ai/projects/wizard_of_wikipedia/",                 "train.json → data/wizard/"),
]


def _download_file(url: str, dest: Path, description: str) -> bool:
    """Download url to dest with progress display.

    Automatically injects Authorization: Bearer {HF_TOKEN} for huggingface.co
    URLs when the env var is set. If auth fails or token is absent, logs a
    clear message and returns False without crashing the rest of the pipeline.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {description}...")

    headers: dict[str, str] = {"User-Agent": "MemTuner/0.0.1"}
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    is_hf_url = "huggingface.co" in url
    if is_hf_url and hf_token and not hf_token.startswith("hf_your"):
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        req = urllib.request.Request(url, headers=headers)
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
    except urllib.error.HTTPError as e:
        if e.code == 401 and is_hf_url:
            print("\r    ✗ Auth required — set HF_TOKEN in .env to download from HuggingFace")
        elif e.code == 403 and is_hf_url:
            print("\r    ✗ Access denied — check HF_TOKEN permissions for this dataset")
        else:
            print(f"\r    ✗ HTTP {e.code}: {e.reason} — skipping, continuing with other datasets")
        if dest.exists():
            dest.unlink()
        return False
    except Exception as e:
        print(f"\r    ✗ Failed: {e} — skipping, continuing with other datasets")
        if dest.exists():
            dest.unlink()
        return False


def print_status() -> None:
    print("\n=== Dataset Status ===\n")

    print("AUTO (locomo — downloaded from snap-research/locomo, no conversion needed):")
    f = DATA_DIR / "locomo10.json"
    status = f"✓ {f.stat().st_size // 1024 // 1024} MB" if f.exists() else "✗ missing"
    print(f"  locomo10.json       {status}")
    if f.exists():
        print("    → Run: memtuner study --gold-dataset data/input/locomo10.json --mode full")

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
            print(f"    → Run: memtuner study --gold-dataset {out.relative_to(project_root)} --mode full")

    print("\nMANUAL DOWNLOAD REQUIRED:")
    for name, url, instructions in MANUAL_DOWNLOADS:
        print(f"  {name:20s}  {url}")
        print(f"    Instructions: {instructions}")

    print()


def load_env() -> None:
    """Load .env (HF_TOKEN etc.) without requiring python-dotenv."""
    env_file = project_root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _hf_token_summary() -> None:
    """Tell the user upfront how HuggingFace downloads will behave."""
    token = os.environ.get("HF_TOKEN", "").strip()
    hf_names = [dest.name for dest, url, _ in DOWNLOADS if "huggingface.co" in url]
    if token and not token.startswith("hf_your"):
        print(f"  HF_TOKEN found — authenticated HuggingFace downloads enabled ({len(hf_names)} files).")
    elif hf_names:
        print("  HF_TOKEN not set — HuggingFace downloads will be attempted anonymously.")
        print(f"    Files that may fail without it: {', '.join(hf_names)}")
        print("    Fix: add HF_TOKEN=hf_... to .env (get one at https://huggingface.co/settings/tokens)")


def do_download() -> None:
    print("\n=== Downloading Datasets ===\n")
    _hf_token_summary()
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
    load_env()
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
