# Vocabulary Expansion

## Goal
To identify potentially new tokens important to the domain, and add them to the vocabulary

## Scripts in this folder
 - mine_tokens.py : identifies potential words that should be added to the vocabulary
 - extend_vocabulary.py : adds the identfied tokens
 - validate_tokenizer.py : verifies all selected additional tokens are added to the tokenizer and model vocabulary length is updated

## Subfolders in this folder
- configs : contains parameters related to raw data location, model name etc
- token_stats : This is where the script mine_tokens.py dumps the shortlisted tokens with their stats
- new_tokens : Contains the list of new tokens to be added to the tokenizer
