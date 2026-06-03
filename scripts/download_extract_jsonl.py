import json
import pandas as pd

# From URL
parquet_url = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_00022.parquet"
df = pd.read_parquet(parquet_url)

# Or from local file
# df = pd.read_parquet("data.parquet")

# Convert to JSONL
output_path = "/root/bigben/dataset/test.jsonl"
df = df.head(100)
df.to_json(output_path, orient="records", lines=True, force_ascii=False)

print(f"Saved {len(df)} records to {output_path}")