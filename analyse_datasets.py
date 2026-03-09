import pandas as pd

# Load CSV file
df = pd.read_csv("./perturbed_datasets/GSM8K_RUP.csv")

# Print column names
print(df.iloc[0]["answer"])

