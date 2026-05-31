# Data Layer

## Schema
- Raw CSVs:
    - ASIN:  Identier for an SKU
    - TITLE: ASIN title
    - BRAND_NAME: SKU brand name
    - CATEGORY: ASIN category hierarchy, example - "Baby Products › Diapering › Disposable Diapers"
    - DESCRIPTION: ASIN description
    - BULLETS: Brief list of bullet points describing the product
    - OVERVIEW: Structured metadata with basic ASIN attributes
    - VARIANT_JSON: The varation spec of this ASIN if has other variants
    - IMAGE_URL: Web URL of the item image
    - BSR: List of the Best Seller category name and rank at multiple ASIN category levels
    - cat_1: The Primary category of the ASIN extracted from the field 'CATEGORY'

## Scripts in this folder:
 - dummy_brands.py : List of brands observed in e-commerce data that are junk values
 - extract_dataset : Extract the .csv data from zip file, format the columns as needed

 ## Subfolders in this folder
  - zipped : contains the zipped version of the raw data
  - raw : contains the extracted csv from the zip folder
  - processed : conatims the train and test data separately
  - raw_stats : contains some basic stats of each column in the raw data