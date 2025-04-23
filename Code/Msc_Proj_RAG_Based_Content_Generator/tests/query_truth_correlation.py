# ====================================================================================##
# query-truth correlation script to compute and track query-ground-truth similarity
# ====================================================================================##
# importing required libraries
import ast
import os
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from retrieval.config import GROUND_TRUTH, PROD_PROCESSED_JSON, QUERY_PROD_TEXT
import matplotlib.pyplot as plt
# ====================================================================================##
# configuration: define embedding models and file paths
# ====================================================================================##
model_names = ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"]

# ====================================================================================##
# load_ground_truth: load and filter ground truth data for training split
# ====================================================================================##
def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH)
    df["item_id"] = df["item_id"].str.lower().apply(ast.literal_eval)
    return df[df["split"] == "train"]

# ====================================================================================##
# load_product_metadata: read processed product metadata json
# ====================================================================================##
def load_product_metadata():
    return pd.read_json(PROD_PROCESSED_JSON)

# ====================================================================================##
# map_queries_to_products: match single-item queries to product metadata
# ====================================================================================##
def map_queries_to_products(gt_df, product_data):
    query_data_map = {}
    gt_df_one = gt_df[gt_df["item_id"].apply(lambda x: len(x) == 1)]
    for _, gt_row in gt_df_one.iterrows():
        matching = product_data[product_data["item_id"] == gt_row["item_id"][0]]
        if not matching.empty:
            query_data_map[gt_row["query"]] = matching.iloc[0]["metadata"]
    return query_data_map

# ====================================================================================##
# compute_sim_scores: calculate cosine similarity between queries and products
# ====================================================================================##
def compute_sim_scores(model, query_data_map):
    scores = {}
    for query, product_text in query_data_map.items():
        q_vec = model.encode(query, convert_to_tensor=True)
        p_vec = model.encode(product_text, convert_to_tensor=True)
        scores[query] = util.cos_sim(q_vec, p_vec).item()
    return scores

# ====================================================================================##
# track_similarity_scores: log similarity scores to mlflow
# ====================================================================================##
def track_similarity_scores(model_name, scores):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Query_Ground_truth_similiarity_tracking")
    with mlflow.start_run(run_name=f"Model: {model_name}"):
        mlflow.log_dict(scores, f"{model_name}_query_gt_sim_scores.json")

# ====================================================================================##
# main:  data loading, similarity computation, and distribution plotting
# ====================================================================================##
def main():
    gt_df = load_ground_truth()
    product_data = load_product_metadata()
    query_data_map = map_queries_to_products(gt_df, product_data)

    # Save query-product map if not already saved
    if not os.path.exists(QUERY_PROD_TEXT):
        pd.DataFrame.from_dict(query_data_map, orient="index", columns=["metadata"]).to_json(QUERY_PROD_TEXT)

    model_scores = {}

    for model_name in model_names:
        model = SentenceTransformer(model_name)
        scores = compute_sim_scores(model, query_data_map)
        track_similarity_scores(model_name, scores)
        model_scores[model_name] = scores  # Collect for optional post-analysis



    # Count how many ground truths each query has
    gt_counts = gt_df["item_id"].apply(len)

    # Group into 1, 2, 3, 4, and 5+
    binned = gt_counts.apply(lambda x: str(x) if x <= 4 else "5+")

    # Count occurrences of each bin
    distribution = binned.value_counts().sort_index(key=lambda x: x.map(lambda y: int(y.rstrip('+'))))

    # Plot
    labels = [f"{k} GT{'s' if k != '1' else ''}" for k in distribution.index]
    sizes = distribution.values

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Query Distribution by Number of Ground Truths(GT)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()