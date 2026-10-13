import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

diameter = np.random.normal(10, 0.5, n)
length = np.random.normal(50, 2.5, n)
weight = np.random.normal(20, 2, n)
surface_score = np.random.uniform(30, 100, n)
thread_score = np.random.uniform(30, 100, n)

defective = (
    (np.abs(diameter - 10) > 0.8) |
    (np.abs(length - 50) > 4) |
    (surface_score < 55) |
    (thread_score < 50)
)

df = pd.DataFrame({
    "diameter_mm": diameter,
    "length_mm": length,
    "weight_g": weight,
    "surface_score": surface_score,
    "thread_score": thread_score,
    "quality": np.where(defective, "Fail", "Pass")
})

df.to_csv("industrial_quality.csv", index=False)

print(df.head())
print("\nClass distribution:")
print(df["quality"].value_counts())