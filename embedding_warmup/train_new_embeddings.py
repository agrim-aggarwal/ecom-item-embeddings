"""Warm up new-token embeddings via MLM with all base model parameters frozen.

Only the rows in the embedding matrix corresponding to newly added vocab tokens
receive gradient updates (enforced via a backward hook).

Usage:
    python -m embedding_warmup.train_new_embeddings --config embedding_warmup/configs/warmup_config.yaml
    python -m embedding_warmup.train_new_embeddings --config embedding_warmup/configs/warmup_config.yaml --cat "Baby Products"
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

from utils.training_funcs import (
    LossLoggingCallback,
    NewTokenEvalCallback,
    NewTokenForceMaskCollator,
    build_input_text,
    load_new_token_ids,
    register_new_token_grad_mask,
    select_callback_dataset,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WarmupConfig:
    cat: str
    tokenizer_dir: Path = field(default_factory=lambda: Path("vocab/tokenizer/v1"))
    new_tokens_dir: Path = field(default_factory=lambda: Path("vocab/new_tokens"))
    data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/embedding_warmup_checkpoint"))
    loss_log_file: Path = field(default_factory=lambda: Path("logs/embedding_warmup_loss.jsonl"))
    eval_log_file: Path = field(default_factory=lambda: Path("logs/embedding_warmup_eval.jsonl"))
    eval_sample_results_file: Path = field(default_factory=lambda: Path("logs/embedding_warmup_eval_examples.jsonl"))
    train_split: float = 0.9
    random_seed: int = 42
    callback_set_size: int = 10_000
    max_length: int = 512
    mlm_probability: float = 0.15
    per_device_train_batch_size: int = 8
    eval_steps: int = 64
    save_steps: int = 1000
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    bf16: bool = True
    fp16: bool = False
    dataloader_pin_memory: bool = False


_PATH_FIELDS = {
    "tokenizer_dir", "new_tokens_dir", "data_dir", "processed_dir",
    "checkpoint_dir", "loss_log_file", "eval_log_file", "eval_sample_results_file"
}


def load_config(path: Path, overrides: dict | None = None) -> WarmupConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    for key in _PATH_FIELDS:
        if key in data:
            data[key] = Path(data[key])
    return WarmupConfig(**data)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def _freeze_base_params(model) -> None:
    """Freeze everything; unfreeze only the token embedding weight."""
    for param in model.parameters():
        param.requires_grad_(False)
    model.model.embeddings.tok_embeddings.weight.requires_grad_(True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_and_split(
    cfg: WarmupConfig,
    tokenizer: AutoTokenizer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = cfg.data_dir / f"data_processed_{cfg.cat}.csv"
    print(f"[warmup] loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"[warmup] {len(df):,} rows")

    for col in ("TITLE", "OVERVIEW", "BULLETS"):
        df[col] = df[col].fillna("").astype(str)

    df["_input_text"] = df.apply(
        lambda r: build_input_text(r, tokenizer.sep_token), axis=1
    )

    train_df = df.sample(frac=cfg.train_split, random_state=cfg.random_seed)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(cfg.processed_dir / f"train_{cfg.cat}.csv", index=False)
    test_df.to_csv(cfg.processed_dir / f"test_{cfg.cat}.csv", index=False)
    print(f"[warmup] train={len(train_df):,}  test={len(test_df):,}  -> {cfg.processed_dir}")

    return train_df, test_df


def _make_hf_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    def _tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}

    ds = Dataset.from_dict({"text": df["_input_text"].tolist()})
    return ds.map(_tokenize, batched=True, remove_columns=["text"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(cfg: WarmupConfig) -> None:
    print(f"[warmup] cat={cfg.cat!r}  tokenizer_dir={cfg.tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer_dir))
    model = AutoModelForMaskedLM.from_pretrained(str(cfg.tokenizer_dir))
    model.resize_token_embeddings(len(tokenizer))

    new_token_ids = load_new_token_ids(tokenizer, cfg.new_tokens_dir, cfg.cat)

    _freeze_base_params(model)
    register_new_token_grad_mask(
        model.model.embeddings.tok_embeddings.weight,
        new_token_ids,
    )
    print(f"[warmup] base params frozen; grad mask on {len(new_token_ids)} new token rows")

    train_df, test_df = _load_and_split(cfg, tokenizer)

    callback_df = select_callback_dataset(test_df, tokenizer, new_token_ids, cfg.callback_set_size)
    callback_df.to_csv(cfg.processed_dir / f"callback_{cfg.cat}.csv", index=False)
    print(f"[warmup] callback set: {len(callback_df):,} samples")

    train_ds = _make_hf_dataset(train_df, tokenizer, cfg.max_length)
    callback_ds = _make_hf_dataset(callback_df, tokenizer, cfg.max_length)

    collator = NewTokenForceMaskCollator(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        mlm_probability=cfg.mlm_probability,
    )

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[warmup] device={device}")

    for p in (cfg.loss_log_file.parent, cfg.eval_log_file.parent, cfg.eval_sample_results_file.parent, cfg.checkpoint_dir):
        p.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(cfg.checkpoint_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        save_steps=cfg.save_steps,
        save_total_limit=5,
        logging_steps=1,
        logging_strategy="steps",
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[
            LossLoggingCallback(log_file=cfg.loss_log_file),
            NewTokenEvalCallback(
                model=model,
                tokenizer=tokenizer,
                callback_dataset=callback_ds,
                new_token_ids=new_token_ids,
                log_file=cfg.eval_log_file,
                log_examples_file=cfg.eval_sample_results_file,
                collator=collator,
                eval_steps=cfg.eval_steps,
                device=device,
            ),
        ],
    )

    print("[warmup] starting training")
    trainer.train()
    print("[warmup] done")

    # eval_sample_results_file = 'logs/embedding_warmup_eval_examples.jsonl' # cfg.eval_sample_results_file
    # df = pd.read_json(eval_sample_results_file, lines=True)
    # new_filename = eval_sample_results_file.replace('.jsonl', '.csv')
    # df.to_csv(new_filename, index=False)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm up new token embeddings via MLM.")
    parser.add_argument(
        "--config", type=Path,
        default=Path("embedding_warmup/configs/warmup_config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--cat", type=str, default=None, help="Override category name")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides={"cat": args.cat})
    run(cfg)
