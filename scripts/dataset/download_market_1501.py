# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd
import os

# Set the dataset handle
dataset_handle = "pengcw1/market-1501"

# Download the dataset
print(f"Downloading dataset: {dataset_handle}")
dataset_path = kagglehub.dataset_download(dataset_handle)

print(f"Dataset downloaded to: {dataset_path}")

# List the downloaded files
if os.path.exists(dataset_path):
    print("Downloaded files:")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"  - {file_path}")

    # Try to load data if there are CSV files
    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if csv_files:
        print(f"\nFound {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
                print("    First 5 records:")
                print(df.head())
                print()
            except Exception as e:
                print(f"    Error reading {csv_file}: {e}")
    else:
        print("\nNo CSV files found in the dataset.")
        print("The dataset may contain other file types (images, etc.)")
else:
    print(f"Error: Dataset path {dataset_path} does not exist")