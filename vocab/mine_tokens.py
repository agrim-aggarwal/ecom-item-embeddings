"""Analyse tokenisation of the raw corpus and produce candidate new-token reports.

Pipeline position: step 1 of vocab expansion.

Usage:
    python -m vocab.mine_tokens --cat baby
    python -m vocab.mine_tokens --cat electronics --min-freq 10
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from agrim_modules import create_sheet

_TEXT_COLS = ("TITLE", "CATEGORY", "OVERVIEW", "BULLETS")
constants = yaml.safe_load(open('constants.yaml'))
_TOKENIZER_HUB = constants['raw_model_hf_path'] if constants['model_from_local_or_hf']=='hf' else constants['raw_model_local_path']


@dataclass
class MineConfig:
    cat: str
    corpus_dir: Path = field(default_factory=lambda: Path("data/raw"))
    output_dir: Path = field(default_factory=lambda: Path("vocab/token_stats"))
    tokenizer_name: str = _TOKENIZER_HUB
    text_cols: tuple[str, ...] = _TEXT_COLS
    min_word_freq: int = 5


# ---------------------------------------------------------------------------
# Word extraction
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"'(?:re|ve|ll|[stdm])|[A-Za-z0-9][A-Za-z0-9\.]*[A-Za-z0-9]|[A-Za-z0-9]")
_FLOAT_RE = re.compile(r"^\d+\.\d+$")
_SAMPLE_MAX_LEN = 2000


def _word_counts(series: pd.Series) -> tuple[Counter, dict[str, str]]:
    counts: Counter = Counter()
    samples: dict[str, str] = {}
    for text in series.dropna():
        text = str(text)
        for word in _WORD_RE.findall(text):
            counts[word] += 1
            if word not in samples:
                samples[word] = text[:_SAMPLE_MAX_LEN]
    return counts, samples


# ---------------------------------------------------------------------------
# Token analysis
# ---------------------------------------------------------------------------

def _n_subtokens(word: str, tokenizer) -> int:
    ids = tokenizer.encode(word, add_special_tokens=False)
    return len(ids), ids


def build_reports(
    df: pd.DataFrame,
    tokenizer,
    text_cols: tuple[str, ...],
    min_word_freq: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    col_counts: dict[str, Counter] = {}
    col_samples: dict[str, dict[str, str]] = {}
    for col in text_cols:
        if col in df.columns:
            col_counts[col], col_samples[col] = _word_counts(df[col])

    total_counts: Counter = sum(col_counts.values(), Counter())

    # Merge samples: prefer sample from the first col that has one
    all_samples: dict[str, str] = {}
    for col in reversed(text_cols):
        all_samples.update(col_samples.get(col, {}))

    # Tokenise each unique word once
    candidates = [w for w, freq in total_counts.items() if freq >= min_word_freq and not _FLOAT_RE.match(w)]
    print(f"[mine] tokenising {len(candidates):,} unique words (min_freq={min_word_freq})")

    split_rows = []
    existing_rows = []

    for word in candidates:
        n, subtoken_ids = _n_subtokens(word, tokenizer)
        # import pdb;pdb.set_trace();
        freq_by_col = {col: col_counts.get(col, Counter())[word] for col in text_cols}

        sample = all_samples.get(word, "")
        if n > 1:
            subtokens = [tokenizer.decode(x, clean_up_tokenization_spaces=False) for x in subtoken_ids]
            split_rows.append({
                "word": word,
                "n_subtokens": n,
                "subtokens": ' - '.join(subtokens),
                **{f"freq_{col.lower()}": freq_by_col[col] for col in text_cols},
                "freq_total": total_counts[word],
                "sample_text": sample,
            })
        else:
            existing_rows.append({
                "word": word,
                "n_subtokens": n,
                **{f"freq_{col.lower()}": freq_by_col[col] for col in text_cols},
                "freq_total": total_counts[word],
                "sample_text": sample,
            })

    new_tokens_df = (
        pd.DataFrame(split_rows)
        .sort_values("freq_total", ascending=False)
        .reset_index(drop=True)
    )
    existing_df = (
        pd.DataFrame(existing_rows)
        .sort_values("freq_total", ascending=False)
        .reset_index(drop=True)
    )
    return new_tokens_df, existing_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(cfg: MineConfig) -> None:
    corpus_path = cfg.corpus_dir / f"data_processed_{cfg.cat}.csv"
    print(f"[mine] loading corpus from {corpus_path}")
    df = pd.read_csv(corpus_path, low_memory=False)
    print(f"[mine] {len(df):,} rows")

    print(f"[mine] loading tokenizer: {cfg.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    new_tokens_df, existing_df = build_reports(df, tokenizer, cfg.text_cols, cfg.min_word_freq)
    print(f"[mine] split-token candidates: {len(new_tokens_df):,}")
    print(f"[mine] existing single-token words: {len(existing_df):,}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"token_analysis_{cfg.cat}.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        create_sheet(new_tokens_df, writer, "new_token_candidates")
        create_sheet(existing_df, writer, "existing_tokens")
    print(f"[mine] saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine tokenisation candidates from raw corpus.")
    parser.add_argument("--cat", required=True, help="Category name, e.g. 'baby'")
    parser.add_argument("--min-freq", type=int, default=5, help="Min word frequency to include")
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/raw"),
        help="Directory containing data_processed_{cat}.csv"
    )
    args = parser.parse_args()
    run(MineConfig(cat=args.cat, corpus_dir=args.corpus_dir, min_word_freq=args.min_freq))
