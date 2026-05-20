import pandas as pd

# File paths
file1_path = "/media/sf_Projects/ORR/GPGB/data/base_model_data.xlsx"
file2_path = "/media/sf_Projects/ORR/GPGB/data/acitve_learning_data.xlsx"
file3_path = "/media/sf_Projects/ORR/GPGB/data/Predict_model_Ce_processed_0308_rank_Z0_Z10.xlsx"

# Number of rows to skip when reading Excel files
skiprows = 1

# Row offset used to match the original Excel row numbers
row_offset = 3

# Read Excel files
df1 = pd.read_excel(file1_path, skiprows=skiprows)
df2 = pd.read_excel(file2_path, skiprows=skiprows)
df3 = pd.read_excel(file3_path, skiprows=skiprows)

# Select descriptor columns for duplicate checking
part1 = df1.iloc[:, 5:25]
part2 = df2.iloc[:, 5:25]
part3 = df3.iloc[:, 5:25]

print("Number of base model data:", len(part1))
print("Number of active learning data:", len(part2))
print("Number of prediction data:", len(part3))

# Convert each row into a hash value for fast duplicate checking
def get_row_hash(df):
    return pd.util.hash_pandas_object(df, index=False)


# Check duplicate rows between two datasets
def find_duplicates(a, a_index, b, b_index):
    a_hash = get_row_hash(a)
    b_hash = get_row_hash(b)

    # Build a mapping from hash value to Excel row numbers in dataset b
    b_hash_to_rows = {}

    for j, h in enumerate(b_hash):
        if h not in b_hash_to_rows:
            b_hash_to_rows[h] = []
        b_hash_to_rows[h].append(b_index[j] + row_offset)

    # Check whether rows in dataset a appear in dataset b
    found = False

    for i, h in enumerate(a_hash):
        if h in b_hash_to_rows:
            found = True
            for b_row in b_hash_to_rows[h]:
                print(
                    f"Duplicate found: row {a_index[i] + row_offset} "
                    f"and row {b_row}"
                )

    if not found:
        print("No duplicate rows found.")


# Check duplicate rows within one dataset
def find_internal_duplicates(df):
    row_hash = get_row_hash(df)

    duplicated_mask = row_hash.duplicated(keep=False)

    if not duplicated_mask.any():
        print("No duplicate rows found.")
        return

    duplicated_hash = row_hash[duplicated_mask]

    # Group duplicated rows by hash value
    hash_to_rows = {}

    for idx, h in duplicated_hash.items():
        if h not in hash_to_rows:
            hash_to_rows[h] = []
        hash_to_rows[h].append(idx + row_offset)

    for rows in hash_to_rows.values():
        if len(rows) > 1:
            print(f"Duplicate rows found: {rows}")


print("\nChecking base model data vs active learning data...")
find_duplicates(part1, df1.index, part2, df2.index)

print("\nChecking base model data vs prediction data...")
find_duplicates(part1, df1.index, part3, df3.index)

print("\nChecking active learning data vs prediction data...")
find_duplicates(part2, df2.index, part3, df3.index)

print("\nChecking duplicates within base model data...")
find_internal_duplicates(part1)

print("\nChecking duplicates within active learning data...")
find_internal_duplicates(part2)

print("\nChecking duplicates within prediction data...")
find_internal_duplicates(part3)
