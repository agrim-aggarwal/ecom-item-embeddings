Generate boilerplate for the vocab/ module.

Create the following files with working skeleton code:

- vocab/mine_tokens.py
  - Give an input category {cat}, Load raw corpus from data/raw/data_processed_{cat}.csv
  - Load the ModernBERT tokenizer (answerdotai/ModernBERT-base) from HuggingFace
  - Tokenize the ASIN TITLE, CATEGORY, OVERVIEW and BULLETS separately
  - For each column source, break the text into words using regex and based on the criteria explain below. Then words that are being split into more than one tokens by the tokenizer and create 'new tokens report' in below format. These are candidates for adding to the vocabulary
    - Word
    - No. of tokens being split into
    - Word Frequency in TITLE in the whole corpus
    - Word Frequency in CATEGORY in the whole corpus
    - Word Frequency in OVERVIEW in the whole corpus
    - Word Frequency in BULLETS in the whole corpus
  - The criteria for a word
    - Text with apostrophe should be split into two words (eg. don't -> don + 't)
    - Hyphenated text should be treated as two separate words
  - Exclude words that look like floating point numbers (eg. 5.7) from the new tokens report
  - Create an 'existing token report' to capture highest frequency words that are not being split into tokens
  - Save both these reports to this excel file using agrim_modules.create_sheet function in two separate tabs: vocab/token_stats/token_analysis_{cat}.xlsx

<Then a human user will read the file 'vocab/token_stats/token_analysis_{cat}.xlsx' and select the tokens to be finally added to the vocabulary through a file named: vocab/new_tokens/new_tokens_{cat}.csv>

- vocab/extend_vocabulary.py
  - Load the ModernBERT tokeniser and model
  - Add the new tokens to the tokeniser vocabulary
  - Assert len(tokenizer) == original_size + n_new_tokens
  - Assert all tokens in new_tokens.txt are present
  - Run sample encoding and assert no UNK on known domain terms
  - Save the new tokeniser and the model in the location: vocab/tokenizer/v1

Follow conventions in CLAUDE.md. Use dataclasses for config loading.
