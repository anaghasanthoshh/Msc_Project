import ast
import mlflow
import pandas as pd
import retrieval.config as config
from retrieval.data_retrieval import Retrieval
from utils.utils import print_banner

ground_truth_df = pd.read_csv(config.GROUND_TRUTH)  # Adjust path as needed

# select only the train splits from the query set:
ground_truth_df = ground_truth_df[ground_truth_df['split'] == 'train']
ground_truth_df['item_id'] = ground_truth_df['item_id'].apply(lambda x: [doc.lower() for doc in ast.literal_eval(x)])

# Convert ground truth into a dictionary (query : list of relevant product IDs)
ground_truth_dict = {
    row["query"]: row["item_id"]  # Convert comma-separated IDs to a list
    for _, row in ground_truth_df.iterrows()
}


def evaluate_retrieval(retrieved_results, k=5, experiment_name="retrieval_experiment", run_name="default_run"):


    results = []
    for query, retrieved_ids in retrieved_results.items():
        groundtruth_docs = ground_truth_dict.get(query, [])

        if not groundtruth_docs:

            return
            # Skip queries with no ground truth
        print_banner("Analysing the data:\n", f"User Query : {query}",
                     f"Ground truth IDs : {groundtruth_docs}",
                     f"No of Ground truth IDs : {len(groundtruth_docs)}",
                     f"First (k){k} Ids : {retrieved_ids}")

        # Compute relevance scores (1 if in ground truth, 0 otherwise)
        relevance_scores = [1 if doc in groundtruth_docs else 0 for doc in retrieved_ids[:k]]
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)  # Best possible ranking
        print_banner(f"Retrieval Evaluation results :\n",
                     f"Relevance scores : {relevance_scores}")

        #  Precision@K: Fraction of retrieved items that are relevant
        precision_k = sum(relevance_scores) / k

        # Recall@K: Fraction of total ground truth products retrieved within top-K
        recall_k = sum(relevance_scores) / len(groundtruth_docs) if groundtruth_docs else 0

        # Store results
        results.append({"query" : query, "k_value" : k, f"precision_{k}" : precision_k, f"recall_{k}" : recall_k})

        # Log metrics in MLflow


    return pd.DataFrame(results)


if __name__ == "__main__":
    ret = Retrieval()
    query_text = input("Enter your query :")
    query_embed = ret.query_to_vector(query_text)
    gen, product_results = ret.data_retrieval(query_text, query_embed, "l2")

    # Run evaluation and log to MLflow
    evaluation_results = evaluate_retrieval(product_results, k=10, run_name="baseline_retrieval")
    print(f"{evaluation_results}\n")
