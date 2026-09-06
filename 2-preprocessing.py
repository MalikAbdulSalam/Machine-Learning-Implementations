import pandas as pd
# to read our csv file
df = pd.read_csv("data2.csv")
print(type(df))

basket = df["Name"]
#print(basket)


#print(df["Gender"])

print(type(basket))