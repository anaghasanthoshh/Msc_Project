import ast
import os
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from retrieval.config import GROUND_TRUTH, PROD_PROCESSED_JSON, QUERY_PROD_TEXT

model_names = ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"]

def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH)
    df["item_id"] = df["item_id"].str.lower().apply(ast.literal_eval)
    return df[df["split"] == "train"]

def load_product_metadata():
    return pd.read_json(PROD_PROCESSED_JSON)

def map_queries_to_products(gt_df, product_data):
    query_data_map = {}
    gt_df_one = gt_df[gt_df["item_id"].apply(lambda x: len(x) == 1)]
    for _, gt_row in gt_df_one.iterrows():
        matching = product_data[product_data["item_id"] == gt_row["item_id"][0]]
        if not matching.empty:
            query_data_map[gt_row["query"]] = matching.iloc[0]["metadata"]
    return query_data_map

def compute_sim_scores(model, query_data_map):
    scores = {}
    for query, product_text in query_data_map.items():
        q_vec = model.encode(query, convert_to_tensor=True)
        p_vec = model.encode(product_text, convert_to_tensor=True)
        scores[query] = util.cos_sim(q_vec, p_vec).item()
    return scores

def track_similarity_scores(model_name, scores):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Query_Ground_truth_similiarity_tracking")
    with mlflow.start_run(run_name=f"Model: {model_name}"):
        mlflow.log_dict(scores, f"{model_name}_query_gt_sim_scores.json")

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

if __name__ == "__main__":
    main()