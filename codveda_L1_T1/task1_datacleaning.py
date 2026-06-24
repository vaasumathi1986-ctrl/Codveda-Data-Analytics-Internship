import pandas as pd

# Load Dataset
df = pd.read_csv("churn-bigml-80.csv")
print(df.head())

print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Save Cleaned Dataset
df.to_csv("cleaned_churn_dataset.csv", index=False)

print("\nData Cleaning Completed Successfully!\n")