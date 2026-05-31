Generate boilerplate for the vocab/ module.

Create the following files with working skeleton code:

- embedding_warmup/train_new_embeddings.py
  - Read the model and the tokenizer from the folder: vocab/tokenizer/v1
  - Freeze all base parameters of the model. For the embedding layers, register a backward hook or zero-gradient mask ensuring ONLY the token IDs corresponding to the newly added vocabulary parameters receive weight updates during gradient steps.
  - Load the data from data/raw/data_processed_{cat}.csv, ensuring NaN values are safely handled as empty strings.
  - Split it into train-test based on a fixed random seed so that it is split in the same way every time.
  - Save the respective train and test data sets in the folder data/processed
  - Also carve another 'callback' data set of size 10000 randomly out of the testing set on which the model performance (accuracy + loss) can be evaluated after every 64 steps later on using Trainer Callbacks. The data said should be selected in a way that it has as many new tokens as possible.
  - Prepare the input for each ASIN dynamically using the tokenizer's native special tokens: "Title: {title} {tokenizer.sep_token} Attributes: {overview} {tokenizer.sep_token} Features: {bullets}"
  - Use Hugging Face's DataCollatorForLanguageModeling configured for 15% random masking probability to handle the Masked Language Modeling task. Please note that all the newly added tokens should be masked, so they will be part of the 15%
  - Create and run a HuggingFace Trainer using the MLM task.
  - Print and append the training loss after every step into a log file so that it is viewable using a streamlit dashboard.
  - Also log the model inputs, outputs and accuracy on the callback set into a separate file again, viewable using a streamlit dashboard.
  - Checkpoint the model after every 1000 steps into the folder models/embedding_warmup_checkpoint/
  - Make sure to put any resuable components in utils/training_funcs.py instead of directly putting in embedding_warmup/train_new_embeddings.py