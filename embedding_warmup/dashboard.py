"""Streamlit dashboard for monitoring embedding warmup training.

Usage:
    streamlit run embedding_warmup/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOSS_LOG = Path("logs/embedding_warmup_loss.jsonl")
EVAL_LOG = Path("logs/embedding_warmup_eval.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


st.set_page_config(page_title="Embedding Warmup", layout="wide")
st.title("Embedding Warmup — Training Dashboard")

col_left, col_right = st.columns(2)

# ---- Training loss ----
with col_left:
    st.subheader("Training Loss (per step)")
    loss_records = _read_jsonl(LOSS_LOG)
    if loss_records:
        df_loss = pd.DataFrame(loss_records)
        st.line_chart(df_loss.set_index("step")["loss"])
        st.dataframe(df_loss.tail(20), use_container_width=True)
    else:
        st.info(f"Waiting for `{LOSS_LOG}` …")

# ---- Callback-set eval ----
with col_right:
    st.subheader("Callback Set — Accuracy & Loss (every 64 steps)")
    eval_records = _read_jsonl(EVAL_LOG)
    if eval_records:
        df_eval = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "samples"} for r in eval_records]
        )
        st.line_chart(df_eval.set_index("step")[["accuracy", "new_token_accuracy"]])
        st.line_chart(df_eval.set_index("step")["loss"])
        st.dataframe(df_eval.tail(10), use_container_width=True)
    else:
        st.info(f"Waiting for `{EVAL_LOG}` …")

# ---- Sample predictions ----
st.subheader("Sample Predictions (latest eval checkpoint)")
if eval_records:
    latest = eval_records[-1]
    st.caption(f"Step {latest['step']} — accuracy={latest['accuracy']}  new_token_acc={latest['new_token_accuracy']}")
    for sample in latest.get("samples", []):
        with st.expander(sample["input"][:120] + "…"):
            st.markdown(f"**Target:** `{sample['target']}`")
            st.markdown(f"**Predicted:** `{sample['predicted']}`")
