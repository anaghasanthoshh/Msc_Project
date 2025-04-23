import pandas as pd
from retrieval.config import SIM_SCORES
import ast
import os
import matplotlib.pyplot as plt

with open(os.path.join(SIM_SCORES,'query_length_precision.txt'),"r") as file:
    data = file.read()

query_metrics=ast.literal_eval(data)
df = pd.DataFrame.from_dict(query_metrics, orient="index").reset_index()
df = df.rename(columns={"index": "query_id"})
print(df.head())
df["query_id"] = df["query_id"].astype(int)
corr=df['length'].corr(df['precision'])
print(f"Correlation:{corr}" )
# Correlation:0.17160190736722847

#Plotting
plt.figure(figsize=(8, 5))
plt.scatter(df["length"], df["precision"], color="blue", alpha=0.5)
plt.title("Scatter Plot: Query Length vs Precision")
plt.xlabel("Query Length")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()
plt.show()

