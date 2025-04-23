# ====================================================================================##
# automated semantic evaluation script combining retrieval and metrics logging
# ====================================================================================##
# importing required libraries
from retrieval.config import GROUND_TRUTH
from sentence_transformers import SentenceTransformer
from retrieval.data_retrieval import Retrieval,Metric
from retrieval.retrieval_eval import evaluate_retrieval
from utils.utils import log_timing
import pandas as pd
import mlflow


# ====================================================================================##
# mlflow setup: tracking server and experiment initialization
# ====================================================================================##
# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Test_Semantic_retrieval_and_evaluation")

# ====================================================================================##
# initialization: load embedding model, set retrieval parameters
# ====================================================================================##
## setting initial values :
model_name="all-MiniLM-L6-v2"
#model_name="multi-qa-mpnet-base-dot-v1"
model=SentenceTransformer(model_name)
kr=20 # Total No.of items to be retrieved
k = 3  # K for Precision/Recall@k
stype = Metric.L2  # similiarity type
with mlflow.start_run(run_name=f"{stype}_Similiarity_Model:{model_name}_K:{k}") as run:

    ret = Retrieval(model,kr)
    # query_text = input("Enter your query :")
    ground_truth_path = GROUND_TRUTH
    gt_df = pd.read_csv(ground_truth_path)
    gt_df=gt_df[gt_df["split"]=="test"]
    all_results = []
    all_precisions = []
    all_recalls = []
    all_mrr = []
    query_length_precision={}

    # ====================================================================================##
    # evaluation loop: iterate over test queries and compute metrics
    # ====================================================================================##
    for idx, row in gt_df.iterrows():
        query_text = row["query"]
        query_len=len(row["query"])
        with log_timing(f"Query_Embedding_{idx}"):
            query_embed = ret.query_to_vector(query_text)
        with log_timing(f"Data_Retrieval_{idx}"):
            _, product_results = ret.data_retrieval(query_text, query_embed, type=stype)
        product_results={query_text:product_results[query_text][:k]}
        print(f"new prod={product_results}")
        eval_result = evaluate_retrieval(
                product_results,
                k=k,
                run_name=f"retrieval_l2_query_{idx}"
            )
        all_results.append({"query": query_text, "metrics": eval_result})
        all_precisions.append(eval_result[f"precision_{k}"])
        all_recalls.append(eval_result[f"recall_{k}"])
        all_mrr.append(eval_result["MRR_score"])
        query_length_precision[idx] = {"length": query_len, "precision": (eval_result[f"precision_{k}"]).iloc[0]}


        mlflow.log_param(f"K_value", k)
        mlflow.log_param("Total Retrieved",kr)
        mlflow.log_param("model",model_name)
        mlflow.log_param(f"query_length{idx}", query_len)
        mlflow.log_metric(f"Precision_K{idx}", eval_result[f"precision_{k}"])
        mlflow.log_metric(f"Recall_K_{idx}", eval_result[f"recall_{k}"])
        mlflow.log_metric("MRR_score", eval_result["MRR_score"])



        print(f"Query {idx}: {query_text}\nMetrics: {eval_result}\n")
    # ====================================================================================##
    # calculate and log average precision, recall, and MRR
    # ====================================================================================##
    avg_precision = (sum(all_precisions)/len(all_precisions)).iloc[0]
    avg_recall = (sum(all_recalls)/len(all_recalls)).iloc[0]
    avg_mrr=(sum(all_mrr)/len(all_mrr)).iloc[0]
    mlflow.log_metric("Average_Precision",avg_precision)
    mlflow.log_metric("Average_Recall",avg_recall)
    mlflow.log_metric("Average_MRR_score", avg_mrr)
    mlflow.log_dict(query_length_precision,"query_length_precision")
    print(f"Average recall: {avg_recall}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average MRR :{avg_mrr}")


