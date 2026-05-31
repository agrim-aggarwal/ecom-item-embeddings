"""Extend ModernBERT tokenizer with domain-specific tokens and save the result.

Pipeline position: step 4 of vocab expansion (after manual review of new_tokens_{cat}.csv).

Usage:
    python -m vocab.extend_vocabulary --cat "Baby products" --tokenizer-out-dir "vocab/tokenizer/v1"
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

_TOKENIZER_HUB = "answerdotai/ModernBERT-base"


@dataclass
class ExtendConfig:
    cat: str
    tokenizer_out_dir: Path
    new_tokens_dir: Path = field(default_factory=lambda: Path("vocab/new_tokens"))
    model_name: str = _TOKENIZER_HUB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_new_tokens(cfg: ExtendConfig) -> list[str]:
    tokens_path = cfg.new_tokens_dir / f"new_tokens_{cfg.cat}.csv"
    if not tokens_path.exists():
        raise FileNotFoundError(f"New tokens file not found: {tokens_path}")
    import pandas as pd
    df = pd.read_csv(tokens_path)
    col = df.columns[0]
    tokens = df[col].dropna().str.strip().tolist()
    return [t for t in tokens if t]


def _mean_embedding_init(model, n_new: int) -> None:
    with torch.no_grad():
        embeddings = model.model.embeddings.tok_embeddings.weight
        mean_vec = embeddings[:-n_new].mean(dim=0)
        embeddings[-n_new:] = mean_vec


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def _assert_vocab_size(tokenizer, original_size: int, n_new: int) -> None:
    actual = len(tokenizer)
    expected = original_size + n_new
    assert actual == expected, (
        f"Vocab size mismatch: expected {expected}, got {actual}"
    )
    print(f"[extend] vocab size check passed: {original_size} + {n_new} = {actual}")


def _assert_tokens_present(tokenizer, new_tokens: list[str]) -> None:
    missing = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    assert not missing, f"Tokens missing from vocab: {missing[:10]}"
    print(f"[extend] all {len(new_tokens)} new tokens present in vocab")


def _assert_no_unk(tokenizer, new_tokens: list[str]) -> None:
    unk_id = tokenizer.unk_token_id
    failures = []
    for token in new_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if unk_id in ids:
            failures.append(token)
    assert not failures, f"UNK produced for domain tokens: {failures[:10]}"
    print(f"[extend] no UNK tokens found for {len(new_tokens)} domain terms")


def _assert_single_token(tokenizer, new_tokens: list[str]) -> None:
    multi = [t for t in new_tokens if len(tokenizer.encode(t, add_special_tokens=False)) != 1]
    assert not multi, f"New tokens not encoded as single token: {multi[:10]}"
    print(f"[extend] all {len(new_tokens)} new tokens encode as single tokens")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(cfg: ExtendConfig) -> None:
    new_tokens = _load_new_tokens(cfg)
    print(f"[extend] loaded {len(new_tokens)} new tokens for category '{cfg.cat}'")

    tokenizer_src = _TOKENIZER_HUB

    print(f"[extend] loading tokenizer from {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    original_size = len(tokenizer)
    print(f"[extend] original vocab size: {original_size}")

    # Filter tokens already in vocab
    already_present = [t for t in new_tokens if t in tokenizer.get_vocab()]
    if already_present:
        print(f"[extend] skipping {len(already_present)} tokens already in vocab")
    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    n_new = len(tokens_to_add)
    print(f"[extend] adding {n_new} new tokens")

    tokenizer.add_tokens(tokens_to_add)

    print(f"[extend] loading model from {tokenizer_src}")
    model = AutoModelForMaskedLM.from_pretrained(tokenizer_src)
    model.resize_token_embeddings(len(tokenizer))

    if n_new > 0:
        _mean_embedding_init(model, n_new)
        print(f"[extend] initialised {n_new} new embeddings to mean of existing matrix")

    # Assertions
    _assert_vocab_size(tokenizer, original_size, n_new)
    _assert_tokens_present(tokenizer, tokens_to_add)
    _assert_no_unk(tokenizer, tokens_to_add)
    _assert_single_token(tokenizer, tokens_to_add)

    cfg.tokenizer_out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(cfg.tokenizer_out_dir)
    model.save_pretrained(cfg.tokenizer_out_dir)
    print(f"[extend] saved tokenizer + model -> {cfg.tokenizer_out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extend ModernBERT vocab with domain tokens.")
    parser.add_argument("--cat", required=True, help="Category name, e.g. 'baby'")
    parser.add_argument(
        "--new-tokens-dir", type=Path, default=Path("vocab/new_tokens"),
        help="Directory containing new_tokens_{cat}.csv"
    )
    parser.add_argument(
        "--tokenizer-out-dir", type=Path,
        help="Directory containing new tokenizer and model"
    )
    args = parser.parse_args()
    run(ExtendConfig(
        cat=args.cat,
        tokenizer_out_dir=args.tokenizer_out_dir,
        new_tokens_dir=args.new_tokens_dir
    ))
