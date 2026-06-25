import pandas as pd
from pathlib import Path

# Root directory (the folder where this script is located)
ROOT_DIR = Path(__file__).parent.resolve()

# Create output folder
OUTPUT_DIR = ROOT_DIR / "merged_output"
OUTPUT_DIR.mkdir(exist_ok=True)

all_data = []

print(f"Scanning folders in: {ROOT_DIR}\n")

# Loop through each year folder
for year_folder in sorted(ROOT_DIR.iterdir()):
    if not year_folder.is_dir():
        continue

    # Skip output folder
    if year_folder.name == "merged_output":
        continue

    csv_files = sorted(year_folder.glob("*.csv"))

    if not csv_files:
        print(f"Skipping {year_folder.name} (no CSV files found)")
        continue

    print(f"Processing {year_folder.name}...")

    yearly_data = []

    for csv_file in csv_files:
        try:
            print(f"   Reading {csv_file.name}")

            df = pd.read_csv(csv_file)

            # Add metadata columns
            df["Year"] = year_folder.name
            df["Source_File"] = csv_file.name

            yearly_data.append(df)

        except Exception as e:
            print(f"   Error reading {csv_file.name}: {e}")

    if yearly_data:
        # Merge all files for the year
        year_df = pd.concat(yearly_data, ignore_index=True)

        yearly_output = OUTPUT_DIR / f"{year_folder.name}_merged.csv"
        year_df.to_csv(yearly_output, index=False)

        print(f"   Saved {yearly_output.name} ({len(year_df):,} rows)\n")

        all_data.append(year_df)

# Merge all years into one master file
if all_data:
    print("Creating master consolidated file...")

    master_df = pd.concat(all_data, ignore_index=True)

    master_output = OUTPUT_DIR / "startup_consolidated_2015_2021.csv"
    master_df.to_csv(master_output, index=False)

    print("\n===================================")
    print("MERGE COMPLETED SUCCESSFULLY")
    print("===================================")
    print(f"Master file: {master_output}")
    print(f"Total rows: {len(master_df):,}")
    print(f"Total columns: {len(master_df.columns)}")
else:
    print("No CSV files were found.")