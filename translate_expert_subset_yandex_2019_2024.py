import os
import time
import random
import argparse
import pandas as pd
import translators as ts
from tqdm import tqdm

# ---------- CSV SAFETY HELPER ----------
import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def clean_for_csv(x):
    """
    Remove problematic control characters that can break CSV writing on Windows.
    Keeps: \t (0x09), \n (0x0A), \r (0x0D).
    Removes: NUL and other control chars.
    """
    if x is None:
        return ""
    s = str(x)
    return _CONTROL_CHARS.sub("", s)



# ---------- CONFIG ----------
INPUT_FILE = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input\lenta_news_1999_2024_subset.csv"

OUTPUT_TEMPLATE = r"C:\lenta_translate\lenta_news_1999_2024_subset.translated_shard{shard_id}.csv"
ERROR_LOG_TEMPLATE = r"C:\lenta_translate\lenta_news_1999_2024_subset.errors_shard{shard_id}.csv"

# Translate ONLY these columns
COLUMNS_TO_TRANSLATE = ["title", "text"]

# Language settings
SRC_LANG = "ru"
TGT_LANG = "en"

# Chunking parameters
CHUNKSIZE = 400
SLEEP_BETWEEN_ROWS = 0.2    # small random delay added inside translation fn
SLEEP_BETWEEN_CHUNKS = 60   # take long breaks between chunks

# Retries per row
MAX_RETRIES = 5



# ---------- TRANSLATION (YANDEX-ONLY) ----------
def translate_with_retry(text, shard_id, row_id, col_name, error_log_file,
                         max_retries=MAX_RETRIES, base_sleep=2.0):
    """
    Translate using YANDEX only.
    Retries with exponential backoff + random jitter.
    """
    original_text = "" if text is None else str(text)
    if not original_text.strip():
        return ""

    for attempt in range(1, max_retries + 1):
        try:
            # Yandex translation
            translated = ts.translate_text(
                original_text,
                translator="yandex",
                from_language=SRC_LANG,
                to_language=TGT_LANG
            )

            # jitter to avoid being rate-limited
            time.sleep(SLEEP_BETWEEN_ROWS + random.random() * 0.2)
            return translated

        except Exception as e:
            print(
                f"\n[Yandex][shard {shard_id}] error (attempt {attempt}/{max_retries}) "
                f"on row {row_id}, col {col_name}: {e}"
            )
            sleep_time = base_sleep * attempt + random.random()
            print(f"[shard {shard_id}] sleeping {sleep_time:.1f}s before retry...")
            time.sleep(sleep_time)

    # Failed after all retries — log
    print(f"  -> [shard {shard_id}] FAILED row {row_id}, col {col_name}. Logging failure.")
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(f"{row_id},{col_name}\n")

    snippet = original_text[:200].replace("\n", " ")
    return f"[TRANSLATION_FAILED] {snippet}"


# ---------- RESUME SUPPORT ----------
def get_max_done_row(output_file):
    """
    Reads existing output and returns highest orig_row_id finished.
    If no file exists, returns -1.
    """
    if not os.path.exists(output_file):
        return -1
    try:
        df = pd.read_csv(output_file, usecols=["orig_row_id"])
        if df.empty:
            return -1
        return int(df["orig_row_id"].max())
    except Exception:
        return -1


# ---------- MAIN ----------
def main(shard_id, num_shards):
    output_file = OUTPUT_TEMPLATE.format(shard_id=shard_id)
    error_log_file = ERROR_LOG_TEMPLATE.format(shard_id=shard_id)

    # Create error log with header if not present
    if not os.path.exists(error_log_file):
        with open(error_log_file, "w", encoding="utf-8") as f:
            f.write("row_id,column\n")

    # Resume support
    max_done = get_max_done_row(output_file)
    print(f"[shard {shard_id}] Max already translated orig_row_id: {max_done}")

    first_chunk = (max_done < 0)
    total_seen = 0

    # Process input in chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE)):
        chunk_start = total_seen
        chunk_end = chunk_start + len(chunk)
        total_seen += len(chunk)

        print(f"\n[shard {shard_id}] Processing chunk {chunk_idx} (rows {chunk_start}-{chunk_end})")

        df = chunk.reset_index(drop=True)
        df["orig_row_id"] = list(range(chunk_start, chunk_end))

        # Only rows belonging to this shard
        mask = [(rid % num_shards) == shard_id for rid in df["orig_row_id"]]
        shard_df = df[mask].copy()

        if shard_df.empty:
            print(f"[shard {shard_id}] No rows for this shard in this chunk.")
            continue

        # Resume: remove rows already completed
        if max_done >= 0:
            shard_df = shard_df[shard_df["orig_row_id"] > max_done]
            if shard_df.empty:
                print(f"[shard {shard_id}] Rows already translated — skipping chunk.")
                continue

        print(f"[shard {shard_id}] Will translate {len(shard_df)} rows in this chunk")

        # Translate requested columns
        for col in COLUMNS_TO_TRANSLATE:
            print(f"[shard {shard_id}]   Translating column: {col}")

            translated_col = []
            for val, rid in tqdm(zip(shard_df[col].tolist(), shard_df["orig_row_id"].tolist()),
                                 total=len(shard_df), unit="rows"):

                translated = translate_with_retry(
                    val,
                    shard_id=shard_id,
                    row_id=rid,
                    col_name=col,
                    error_log_file=error_log_file,
                )
                translated_col.append(translated)

            shard_df[col + "_en"] = translated_col

        # Write output
        mode = "w" if first_chunk else "a"
        header = first_chunk

        # Clean all object columns before writing
        obj_cols = shard_df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            shard_df[c] = shard_df[c].map(clean_for_csv)

        shard_df.to_csv(
            output_file,
            mode=mode,
            header=header,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        first_chunk = False

        # Update resume pointer
        max_done = int(shard_df["orig_row_id"].max())

        # Pause between chunks to avoid rate limits
        print(f"[shard {shard_id}] Sleeping {SLEEP_BETWEEN_CHUNKS}s...")
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    print(f"\n[shard {shard_id}] ✅ FINISHED.")
    print(f"[shard {shard_id}] Output: {output_file}")
    print(f"[shard {shard_id}] Errors: {error_log_file}")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate expert subset using Yandex-only translators with sharding.")
    parser.add_argument("--shard-id", type=int, required=True, help="Worker shard ID: 0 or 1")
    parser.add_argument("--num-shards", type=int, default=2, help="Total shards (default 2)")

    args = parser.parse_args()
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("shard-id must be between 0 and num_shards-1")

    main(args.shard_id, args.num_shards)
