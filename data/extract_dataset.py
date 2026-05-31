"""Extract, clean, and profile raw e-commerce category data.

Usage:
    python -m data.extract_dataset --cat baby
    python -m data.extract_dataset --cat electronics --source-dir /custom/path
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
import os
import pandas as pd
import unicodedata
import re

from data.dummy_brands import DUMMY_BRANDS

from agrim_modules import create_sheet

# _SOURCE_DIR = Path("/Users/agrimaggarwal/Documents/data/ecom-amazon-pdp")
_SOURCE_DIR = Path("data/zipped")
_RAW_DIR = Path("data/raw")
_STATS_DIR = Path("data/raw_stats")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def copy_and_extract(cat: str, source_dir: Path = _SOURCE_DIR) -> Path:
    """Copy the zip for *cat* into data/raw/ and return the path to the extracted CSV."""
    src = source_dir / f"data_{cat}.zip"
    dest_zip = _RAW_DIR / f"data_{cat}.zip"
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dest_zip)

    with zipfile.ZipFile(dest_zip) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV file found inside {dest_zip}")
        zf.extractall(_RAW_DIR)
        os.remove(dest_zip)
        return _RAW_DIR / csv_names[0]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_category(df: pd.DataFrame) -> pd.DataFrame:
    df["CATEGORY"] = df["CATEGORY"].str.replace("›", ">", regex=False)
    return df


def clean_brand(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["BRAND_NAME"].isin(DUMMY_BRANDS)
    df.loc[mask, "BRAND_NAME"] = ""
    return df


def _parse_bsr_entry(raw: str) -> dict[str, int]:
    """Convert a BSR JSON string to {category_name: rank} mapping."""
    if not raw or pd.isna(raw):
        return {}
    try:
        entries = json.loads(raw)
        return {
            entry["category"][0]: entry["rank"]
            for entry in entries
            if entry.get("category")
        }
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {}


def clean_bsr(df: pd.DataFrame) -> pd.DataFrame:
    df["BSR"] = df["BSR"].apply(_parse_bsr_entry)
    return df

# def correct_encoding_string(text: str) -> str:
#     # This is to handle corruption of certain strings like "𝗢𝗘𝗞𝗢-𝗧𝗘𝗫®" to "\uD835\uDDE2\uD835\uDDD8\uD835\uDDDE\uD835\uDDE2-\uD835\uDDE7\uD835\uDDD8\uD835\uDDEB®"
#     # Convert surrogates first so Python knows their actual character ranges
#     stage1_decoded = text.encode('utf-16', 'surrogatepass').decode('utf-16')

#     # Step 2: Normalize the stylized font back into normal standard characters
#     # 'NFKC' stands for Normalization Form Compatibility Composition
#     recovered_text = unicodedata.normalize('NFKC', stage1_decoded)
#     return recovered_text

def remove_unrecoverable_encoding(text):
    clean_text = re.sub(r'\S*\\u[0-9a-fA-F]{4}\S*', '', text)
    return(clean_text)

def correct_encoding(df: pd.DataFrame, cols : list[str]) -> pd.DataFrame:
    for col in cols:
        mask = df[col].notna() & (df[col] != "")
        df.loc[mask, col] = df.loc[mask, col].apply(remove_unrecoverable_encoding)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_category(df)
    df = clean_brand(df)
    df = clean_bsr(df)

    cols = [x for x in df.columns if df[x].apply(type).value_counts().index[0].__name__ == 'str']
    print('Correcting encoding for columns: ', cols)
    
    df = correct_encoding(df, cols)
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    total = len(df)

    for col in df.columns:
        # if col == 'BSR':
            # import pdb;pdb.set_trace()
        print(f"Computing stats for column: {col}")
        series = df[col]
        str_series = series.astype(str).replace({"nan": "", "None": ""})
        populated_mask = series.notna() & (str_series != "")

        data_type =  json.dumps(df[col].apply(lambda x: type(x).__name__).value_counts().to_dict())
        n_populated = int(populated_mask.sum())
        n_blank = total - n_populated
        lengths = str_series[populated_mask].str.len()

        if col == 'BSR':
            series = series.apply(lambda x: list(x.keys())).apply(lambda x: x[0] if len(x) > 0 else "")
        most_freq_val = series[populated_mask].mode()
        most_freq = most_freq_val.iloc[0] if not most_freq_val.empty else ""
        most_freq_count = int((series == most_freq).sum()) if most_freq != "" else 0

        records.append({
            "column": col,
            "data_type": data_type,
            "total_rows": total,
            "populated_rows": n_populated,
            "blank_rows": n_blank,
            "pct_populated": round(100 * n_populated / total, 2) if total else 0,
            "distinct_values": int(series.nunique(dropna=True)),
            "max_length": int(lengths.max()) if not lengths.empty else 0,
            "min_length": int(lengths.min()) if not lengths.empty else 0,
            "avg_length": round(float(lengths.mean()), 2) if not lengths.empty else 0,
            "most_frequent_value": str(most_freq)[:200],
            "most_frequent_count": most_freq_count,
            "most_frequent_pct": round(100 * most_freq_count / total, 2) if total else 0,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(cat: str, source_dir: Path = _SOURCE_DIR) -> None:
    print(f"[extract] category={cat}")

    csv_path = copy_and_extract(cat, source_dir)
    print(f"[extract] extracted CSV -> {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[extract] loaded {len(df):,} rows x {len(df.columns)} cols")

    df = clean(df)
    print("[extract] cleaning done")

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _RAW_DIR / f"data_processed_{cat}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[extract] saved processed data -> {out_csv}")

    _STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats = compute_stats(df)
    out_stats = _STATS_DIR / f"data_stats_{cat}.xlsx"
    with pd.ExcelWriter(out_stats, engine='xlsxwriter') as writer:
        create_sheet(stats, writer, f'stats_{cat}')

    print(f"[extract] saved stats -> {out_stats}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and clean a raw category zip.")
    parser.add_argument("--cat", required=True, help="Category name, e.g. 'baby'")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_SOURCE_DIR,
        help="Directory containing the source zip files",
    )
    args = parser.parse_args()
    run(args.cat, args.source_dir)
