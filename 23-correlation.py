import pandas as pd

# read data file
df = pd.read_csv("predictive_maintenance.csv")

# find correlation
print(df.corr(numeric_only=True)["Tool wear [min]"].sort_values())


