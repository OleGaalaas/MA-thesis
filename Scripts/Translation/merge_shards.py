import pandas as pd

# Paths to your shard outputs
shard0_path = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Translated files\lenta_news_1999_2024_subset.translated_shard0.csv"
shard1_path = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Translated files\lenta_news_1999_2024_subset.translated_shard1.csv"


# Output combined file
output_path = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Translated files\lenta_news_1999_2024_translated_all.csv"

print("Loading shards...")
df0 = pd.read_csv(shard0_path)
df1 = pd.read_csv(shard1_path)

print("Combining...")
df_all = pd.concat([df0, df1], ignore_index=True)

print("Sorting by orig_row_id...")
df_all = df_all.sort_values("orig_row_id").reset_index(drop=True)

print("Saving combined dataset...")
df_all.to_csv(output_path, index=False, encoding="utf-8")

print("\nDONE.")
print("Combined file:", output_path)
