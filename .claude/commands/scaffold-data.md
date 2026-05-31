Generate boilerplate for the data/ module.

Create the following files with working skeleton code:

- data/extract_dataset.py
  - Given an imput category {cat} copy the file "/Users/agrimaggarwal/Documents/data/ecom-amazon-pdp/data_{cat}.zip" into the folder data/raw/ and unzip it to extract the .csv file. Each file contains e-commerce SKU level metadata for a broader category. The data format is explained in CLAUDE.MD
  - Replace Special characters like '›' from the field 'CATEGORY' with the standard character '>'
  - Replace the BRAND_NAME with blank string if its value is from the file 'data/dummy_brands.py'
  - Clean the field 'BSR' to make to clean its format
    - Current format: [{"rank":<category 1 rank>,"category":["<category 1 name>"],"category_url":["<category 1 URL>"]},{"rank":category 2 rank,"category":["<category 2 name>"],"category_url":["<category 2 URL>"]}, ...]
    - Required format: {'category 1 name': category 1 rank, 'category 2 name': category 2 rank, ...}
  - Save the dataset with the path and name - "data/raw/data_{cat}.zip"
  - Save the summary statistics of each column containing below info with the path and name - data/raw_stats/data_stats_{cat}.csv
    - No. of rows where value is populated
    - No. of blank rows
    - No. of distinct values
    - Longest value length
    - Most frequent value
    - <Think of other important metrics>
  - Delete the copied zip file and any temp folders created

Follow conventions in CLAUDE.md
