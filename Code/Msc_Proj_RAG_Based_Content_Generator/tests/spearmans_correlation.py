# ====================================================================================##
# spearman correlation script comparing similarity scores between models
# ====================================================================================##
# importing required libraries
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from retrieval.config import SIM_SCORES

# ====================================================================================##
# loading similarity scores from MLflow artefacts
# ====================================================================================##
with open(os.path.join(SIM_SCORES,'minilm_scores.txt'),"r")as d1,open(os.path.join(SIM_SCORES,'qanet_scores.txt'),"r") as d2:
    miniLM_scores = ast.literal_eval(d1.read())
    qanet_scores = ast.literal_eval(d2.read())

queries = list(miniLM_scores.keys())
scores1 = [miniLM_scores[q] for q in queries]
scores2 = [qanet_scores[q] for q in queries]

# ====================================================================================##
# compute spearman rank-order correlation coefficient and p-value
# ====================================================================================##
correlation, p_value = spearmanr(scores1, scores2)
print(f"Correlation of :{correlation}")
print('**********')
print(f"p value :{p_value}")


# ====================================================================================##
# scatter plot with regression trendline of similarity scores
# ====================================================================================##
plt.figure(figsize=(8, 6))
sns.regplot(x=scores1, y=scores2)
plt.title(f"Spearman Correlation of Similarity Scores between MiniLM and QANet(ρ = {correlation:.2f}, p = {p_value:.2e})")
plt.xlabel("MiniLM Similarity Scores(Query–Product Pairs)")
plt.ylabel("QANet Similarity Scores (Query–Product Pairs)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================================================================================##
# prepare dataframe for correlation matrix computation
# ====================================================================================##

# Combine scores into a DataFrame
df_scores = pd.DataFrame({
    'MiniLM': scores1,
    'QANet': scores2
})

# Calculate Spearman correlation matrix
spearman_corr = df_scores.corr(method='spearman')

# ====================================================================================##
# plot heatmap of spearman correlation matrix between models
# ====================================================================================##
plt.figure(figsize=(6, 5))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Spearman Correlation Matrix Between MiniLM and QANet Similarity Scores")
plt.tight_layout()
plt.show()