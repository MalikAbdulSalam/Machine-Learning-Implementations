import pandas as pd

# use absolute path this is not best way
# df = pd.read_csv("D:/Machine_learning/3rd_batch/input_csv/predictive_maintenance.csv")
# print(df)


# load file using relative path
# df = pd.read_csv("predictive_maintenance.csv")
# df = pd.read_csv("./predictive_maintenance.csv")
# print(df)


# df = pd.read_csv("./input_folder/data.csv")
# print(df.head())
# print(df)


df = pd.read_csv("../input_folder/data.csv")
print(df.head())
print(df)