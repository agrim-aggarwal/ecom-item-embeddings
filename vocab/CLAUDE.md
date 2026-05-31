# Vocabulary Expansion

## Goal
Add domain tokens not well-represented in ModernBERT's WordPiece vocab:
brand names, SKU patterns, category-specific jargon, units (e.g. "gsm", "denier").## Pipeline order (must run in this sequence)
1. mine_tokens.py     — frequency-based candidate extraction from raw corpus
2. manual review      — human check of new_tokens.txt before committing
3. extend_vocabulary.py   — Add the new tokens to the tokenizer And adjust model vocabulary length. Save the tokeniser and the model to models/tokenizer/v1/
4. validate_tokenizer.py — regression check

## Embedding init strategy
New token embeddings = mean of all existing embeddings (not random init).

## Don'ts
- Don't add tokens already clean single tokens in base vocab
- Don't add more than ~2000 tokens without re-evaluating embedding init strategy