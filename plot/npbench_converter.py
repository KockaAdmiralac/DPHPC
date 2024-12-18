import json
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def process_csv_to_json(csv_path, benchmark_name):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Ensure the required columns are present
    required_columns = [
        "id", "timestamp", "benchmark", "kind", "domain", "dwarf", "preset",
        "mode", "framework", "version", "details", "validated", "time"
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file is missing required columns.")

    # Group by the necessary attributes
    grouped = df[df['benchmark'] == benchmark_name].groupby([
        "timestamp", "benchmark", "preset", "framework", "details"
    ])

    # Define the base directory for results
    base_dir = Path("../results/np_bench_"+benchmark_name)

    for group_keys, group in grouped:
    # Unpack the group keys (timestamp, benchmark, preset, framework, details)
        timestamp, benchmark, preset, framework, details = group_keys
        times = group["time"]
        stats = {
            "mean": times.mean(),
            "min": times.min(),
            "max": times.max(),
            "median": np.median(times), 
            "std": times.std()
        }

        # Create the JSON data
        json_data = {
            "timing": [
                stats["mean"],
                stats["min"],
                stats["max"],
                stats["median"],
                stats["std"]
            ]
        }

        # Create directory for the framework + details combination
        details = details.replace("-", "_")
        dir_name = f"{framework}_{details}"
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

        # Define the file name and save
        if benchmark_name == "adi":
            file_name = f"{pd.to_datetime(timestamp, unit='s').isoformat()}_TSTEPS15N2{preset}.json"
        else:
            file_name = f"{pd.to_datetime(timestamp, unit='s').isoformat()}_N2{preset}.json"
        file_path = dir_path / file_name

        with open(file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Saved: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV to JSON for a specific benchmark.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("benchmark_name", type=str, help="Name of the kernel to process.")

    args = parser.parse_args()

    process_csv_to_json(args.csv_path, args.benchmark_name)