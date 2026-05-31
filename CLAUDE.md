# ecom-item-embeddings

## Project Objective
This project adapts ModernBERT-base for e-commerce product understanding
through a three-stage pipeline:

1. **Vocabulary expansion** — Mine domain-specific tokens (brand names, SKUs,
   units, category jargon) from the product corpus. Filter out tokens already
   covered by the base vocabulary. Expand the tokenizer and initialise new
   embeddings as the mean of existing ones. Save the expanded tokenizer to
   models/tokenizer/v1/ — this becomes the canonical tokenizer for all
   subsequent stages.

2. **Domain-adaptive pretraining (DAPT)** — Continue MLM training on the
   e-commerce corpus using the expanded tokenizer. Goal is to shift
   ModernBERT's embedding space toward product language before any
   task-specific signal is introduced.

3. **Classification fine-tuning** — Attach a classification head to the
   DAPT checkpoint and fine-tune on labelled (product, category) pairs.
   Primary metric is macro-F1 to account for category imbalance.

These stages are strictly sequential. A model from stage N should never
be used to initialise stage N+2 (i.e. do not fine-tune on raw ModernBERT
weights — always go through DAPT first).

## Repository Structure
├── CLAUDE.md                          ✅ global conventions + vocab rules
├── README.md
├── pyproject.toml
│
├── .claude/commands/
│   ├── scaffold-data.md
│   ├── scaffold-vocab.md
│   ├── scaffold-embedding-warmup.md
│   ├── scaffold-pretrain.md
│   ├── scaffold-finetune.md
│   └── scaffold-tests.md
│
├── data/                       CLAUDE.md + raw/ processed/ splits/
├── vocab/                      CLAUDE.md + mine_tokens.py filter_tokens.py validate_tokenizer.py new_tokens.txt configs/
├── embedding_warmup/           CLAUDE.md + mine_tokens.py filter_tokens.py validate_tokenizer.py new_tokens.txt configs/
├── pretrain/                   CLAUDE.md + dataset.py train.py configs/mlm_config.yaml
├── finetune/                   CLAUDE.md + dataset.py train.py evaluate.py configs/clf_config.yaml
├── models/                     checkpoints/ tokenizer/v1/ registry.py
├── inference/                  predictor.py serve.py
├── tests/                      test_data.py test_vocab.py test_pretrain.py test_finetune.py
├── docs/                       architecture.md data_pipeline.md vocab_decisions.md experiment_log.md
└── notebooks/                  exploration.ipynb