import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from retrieval.config import SIM_SCORES

Folder='/Users/anu_nambiar/Desktop/Final project/'

semantic_df = pd.read_csv(os.path.join(SIM_SCORES,'semantic_metrics.csv'))
semantic_df["K"] = semantic_df["K"].astype(int)

# Set seaborn style
sns.set(style="whitegrid")


metrics = ["Avg Precision", "Avg Recall", "MRR"]
similarities = semantic_df["Similarity"].unique()

# for metric in metrics:
#     for sim in similarities:
#         plt.figure(figsize=(8, 6))
#         data = semantic_df[semantic_df["Similarity"] == sim]
#         sns.barplot(x="K", y=metric, hue="Model", data=data)
#         plt.title(f"{metric} by K for {sim} Similarity")
#         plt.xlabel("K (Top-K Retrieved)")
#         plt.ylabel(metric)
#         plt.legend(title="Model")
#         plt.tight_layout()
#         plt.savefig(os.path.join(Folder, f"{sim}_{metric}.png"))



# Set clean style
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(
    x='K',
    y='Avg Precision',
    hue='Similarity',
    style='Model',
    markers=True,
    dashes=True,
    data=semantic_df
)

# Annotate each point with its precision value
for line in ax.lines:
    for x, y in zip(line.get_xdata(), line.get_ydata()):
        ax.text(x, y + 0.005, f"{y:.2f}", ha='center', va='bottom', fontsize=8, color='black')

# Set plot titles and labels
plt.title("Precision Trends Across K Values")
plt.ylabel("Precision")
plt.xlabel("K (Top-K Retrieved)")
plt.ylim(0.3, 0.66)  # adjust as needed for breathing room
plt.tight_layout()
plt.show()




