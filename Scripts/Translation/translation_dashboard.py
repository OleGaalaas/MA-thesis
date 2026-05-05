import time
import pandas as pd
import streamlit as st
import os

# Paths of each shard
SHARD0 = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input\lenta-ru-news.expert_ru_subset.translated_shard0.csv"
SHARD1 = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input\lenta-ru-news.expert_ru_subset.translated_shard1.csv"

ERROR0 = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input\lenta-ru-news.expert_ru_subset.errors_shard0.csv"
ERROR1 = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input\lenta-ru-news.expert_ru_subset.errors_shard1.csv"

TOTAL_ROWS = 128787

def count_data_rows(path):
    if not os.path.exists(path): return 0
    try:
        return max(0, sum(1 for _ in open(path, "r", encoding="utf-8")) - 1)
    except:
        return 0

st.set_page_config(page_title="Translation Dashboard", layout="wide")

st.title("📊 Real-Time Translation Dashboard")

placeholder = st.empty()

START_TIME = time.time()

while True:
    with placeholder.container():

        rows0 = count_data_rows(SHARD0)
        rows1 = count_data_rows(SHARD1)

        errors0 = count_data_rows(ERROR0)
        errors1 = count_data_rows(ERROR1)

        total_done = rows0 + rows1
        pct = total_done / TOTAL_ROWS * 100

        st.subheader("Progress Overview")
        st.progress(pct / 100)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Shard 0 Rows", rows0)
            st.metric("Shard 0 Errors", errors0)
        with col2:
            st.metric("Shard 1 Rows", rows1)
            st.metric("Shard 1 Errors", errors1)

        st.markdown(f"### Total Progress: **{total_done} / {TOTAL_ROWS}** ({pct:.2f}%)")

        # ETA calculation
        elapsed = time.time() - START_TIME
        if total_done > 0:
            rows_per_sec = total_done / elapsed
            hours_remaining = (TOTAL_ROWS - total_done) / rows_per_sec / 3600
            st.metric("Estimated Hours Remaining", f"{hours_remaining:.1f} hours")
        else:
            st.metric("Estimated Hours Remaining", "Calculating...")

        st.write("⏳ Auto-updates every 5 seconds…")

    time.sleep(5)
