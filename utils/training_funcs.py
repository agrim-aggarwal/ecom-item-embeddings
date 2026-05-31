"""Reusable training components shared across embedding_warmup and future stages."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Input formatting
# ---------------------------------------------------------------------------

def build_input_text(row: dict[str, Any], sep_token: str) -> str:
    title = str(row.get("TITLE", "") or "")
    overview = str(row.get("OVERVIEW", "") or "")
    bullets = str(row.get("BULLETS", "") or "")
    return f"Title: {title} {sep_token} Attributes: {overview} {sep_token} Features: {bullets}"


# ---------------------------------------------------------------------------
# New-token utilities
# ---------------------------------------------------------------------------

def load_new_token_ids(
    tokenizer: AutoTokenizer,
    new_tokens_dir: Path,
    cat: str,
) -> list[int]:
    tokens_path = new_tokens_dir / f"new_tokens_{cat}.csv"
    if not tokens_path.exists():
        raise FileNotFoundError(f"New tokens file not found: {tokens_path}")
    df = pd.read_csv(tokens_path)
    col = df.columns[0]
    tokens = df[col].dropna().str.strip().tolist()
    vocab = tokenizer.get_vocab()
    ids = [vocab[t] for t in tokens if t in vocab]
    print(f"[training_funcs] resolved {len(ids)}/{len(tokens)} new token IDs")
    return ids


def select_callback_dataset(
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    new_token_ids: list[int],
    n: int = 10_000,
) -> pd.DataFrame:
    """Return up to *n* rows from *test_df* ranked by number of new tokens in the encoded text."""
    new_token_set = set(new_token_ids)

    def _count_new(text: str) -> int:
        return sum(1 for tid in tokenizer.encode(text, add_special_tokens=False) if tid in new_token_set)

    df = test_df.copy()
    df["_new_token_count"] = df["_input_text"].apply(_count_new)
    result = (
        df.sort_values("_new_token_count", ascending=False)
        .head(n)
        .drop(columns=["_new_token_count"])
        .reset_index(drop=True)
    )
    return result


# ---------------------------------------------------------------------------
# Custom MLM collator — forces all new-vocab tokens to be masked
# ---------------------------------------------------------------------------

class NewTokenForceMaskCollator(DataCollatorForLanguageModeling):
    """Extends DataCollatorForLanguageModeling so that new tokens are always masked."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        new_token_ids: list[int],
        mlm_probability: float = 0.15,
    ) -> None:
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        self._new_token_ids = set(new_token_ids)
        # Pre-built tensor for vectorized isin — avoids 668-iteration Python loop per batch
        self._new_token_tensor = torch.tensor(sorted(new_token_ids), dtype=torch.long)

    def torch_mask_tokens(
        self,
        inputs: Tensor,
        special_tokens_mask: Tensor | None = None,
        offset_mapping: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask_list = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask_list, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Guarantee every new token position is masked (single vectorised op)
        probability_matrix[torch.isin(inputs, self._new_token_tensor)] = 1.0

        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob), generator=self.generator).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        random_replace_prob_scaled = self.random_replace_prob / (1 - self.mask_replace_prob)
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled), generator=self.generator).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


# ---------------------------------------------------------------------------
# Gradient mask — zero gradients for all rows except the new token rows
# ---------------------------------------------------------------------------

def register_new_token_grad_mask(embedding_weight: Tensor, new_token_ids: list[int]) -> None:
    """Register a backward hook that zeroes all gradient rows except the new token IDs."""
    row_mask = torch.zeros(embedding_weight.shape[0], dtype=torch.float32)
    for tid in new_token_ids:
        row_mask[tid] = 1.0

    def _hook(grad: Tensor) -> Tensor:
        return grad * row_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)

    embedding_weight.register_hook(_hook)


# ---------------------------------------------------------------------------
# Training loss logger
# ---------------------------------------------------------------------------

class LossLoggingCallback(TrainerCallback):
    """Appends training loss to a JSONL file after every logged step."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if not logs or "loss" not in logs:
            return
        record = {
            "step": state.global_step,
            "loss": round(logs["loss"], 6),
            "learning_rate": logs.get("learning_rate"),
            "epoch": round(logs.get("epoch", 0), 4),
        }
        print(f"[loss] step={state.global_step}  loss={logs['loss']:.4f}")
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Callback-set evaluator
# ---------------------------------------------------------------------------

class NewTokenEvalCallback(TrainerCallback):
    """Evaluates accuracy + loss on a callback dataset every *eval_steps* steps."""

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        callback_dataset: Dataset,
        new_token_ids: list[int],
        log_file: Path,
        collator: NewTokenForceMaskCollator,
        eval_steps: int = 64,
        eval_batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.callback_dataset = callback_dataset
        self.new_token_ids = new_token_ids
        self._new_token_tensor: Tensor | None = None  # built lazily on first eval
        self.log_file = log_file
        self.collator = collator
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.device = device

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self._evaluate(state.global_step)

    @torch.no_grad()
    def _evaluate(self, step: int) -> None:
        self.model.eval()

        if self._new_token_tensor is None:
            self._new_token_tensor = torch.tensor(
                self.new_token_ids, dtype=torch.long, device=self.device
            )

        loader = DataLoader(
            self.callback_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            shuffle=False,
        )

        total_loss = 0.0
        all_correct = all_total = 0
        new_correct = new_total = 0
        n_batches = 0
        sample_records: list[dict] = []

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            n_batches += 1

            preds = outputs.logits.argmax(dim=-1)
            masked = labels != -100

            all_correct += (preds[masked] == labels[masked]).sum().item()
            all_total += masked.sum().item()

            new_mask = masked & torch.isin(labels, self._new_token_tensor)
            new_correct += (preds[new_mask] == labels[new_mask]).sum().item()
            new_total += new_mask.sum().item()

            if batch_idx*self.eval_batch_size < 16:
                for i in range(input_ids.shape[0]):
                    masked_pos = labels[i] != -100
                    sample_records.append({
                        "input": self.tokenizer.decode(input_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                        "target": self.tokenizer.decode(labels[i], clean_up_tokenization_spaces=False),
                        "predicted": self.tokenizer.decode(preds[i], clean_up_tokenization_spaces=False),
                    })

        avg_loss = total_loss / n_batches if n_batches else 0.0
        accuracy = all_correct / all_total if all_total else 0.0
        new_token_accuracy = new_correct / new_total if new_total else 0.0

        record = {
            "step": step,
            "loss": round(avg_loss, 6),
            "accuracy": round(accuracy, 4),
            "new_token_accuracy": round(new_token_accuracy, 4),
            "total_masked": all_total,
            "new_token_masked": new_total,
            "samples": sample_records,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"[eval] step={step}  loss={avg_loss:.4f}  "
            f"acc={accuracy:.4f}  new_token_acc={new_token_accuracy:.4f}"
        )

        self.model.train()
