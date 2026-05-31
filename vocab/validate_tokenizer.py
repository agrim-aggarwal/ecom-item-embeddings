"""Regression check for the expanded tokenizer.

Pipeline position: step 4 of vocab expansion (after manual review, before training).

Usage:
    python -m vocab.validate_tokenizer --cat "Baby Products" --tokenizer-dir "vocab/tokenizer/v1"
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


@dataclass
class ValidateConfig:
    cat: str
    tokenizer_dir: Path
    new_tokens_dir: Path = field(default_factory=lambda: Path("vocab/new_tokens"))


def _load_new_tokens(cfg: ValidateConfig) -> list[str]:
    tokens_path = cfg.new_tokens_dir / f"new_tokens_{cfg.cat}.csv"
    if not tokens_path.exists():
        raise FileNotFoundError(f"New tokens file not found: {tokens_path}")
    df = pd.read_csv(tokens_path)
    col = df.columns[0]
    return df[col].dropna().str.strip().tolist()


def run(cfg: ValidateConfig) -> None:
    if not cfg.tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {cfg.tokenizer_dir}. Run extend_vocabulary.py first."
        )

    print(f"[validate] loading tokenizer from {cfg.tokenizer_dir}")
    import pdb;pdb.set_trace();
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer_dir))
    new_tokens = _load_new_tokens(cfg)
    vocab = tokenizer.get_vocab()
    unk_id = tokenizer.unk_token_id

    missing = [t for t in new_tokens if t not in vocab]
    assert not missing, f"Tokens missing from vocab: {missing[:10]}"
    print(f"[validate] all {len(new_tokens)} new tokens present in vocab")

    unk_failures = [
        t for t in new_tokens
        if unk_id in tokenizer.encode(t, add_special_tokens=False)
    ]
    assert not unk_failures, f"UNK produced for: {unk_failures[:10]}"
    print(f"[validate] no UNK tokens for {len(new_tokens)} domain terms")

    multi_token = [
        t for t in new_tokens
        if len(tokenizer.encode(t, add_special_tokens=False)) != 1
    ]
    assert not multi_token, f"Not encoded as single token: {multi_token[:10]}"
    print(f"[validate] all {len(new_tokens)} new tokens encode as single tokens")

    print(f"[validate] PASSED — vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate expanded tokenizer against new tokens list.")
    parser.add_argument("--cat", required=True, help="Category name, e.g. 'Baby Products'")
    parser.add_argument(
        "--tokenizer-dir", type=Path,
        help="Path to the expanded tokenizer"
    )
    parser.add_argument(
        "--new-tokens-dir", type=Path, default=Path("vocab/new_tokens"),
        help="Directory containing new_tokens_{cat}.csv"
    )
    args = parser.parse_args()
    run(ValidateConfig(
        cat=args.cat,
        tokenizer_dir=args.tokenizer_dir,
        new_tokens_dir=args.new_tokens_dir,
    ))
