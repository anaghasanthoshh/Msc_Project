from retrieval.config import GROUND_TRUTH
from retrieval.data_retrieval import Retrieval,Metric
from retrieval.retrieval_eval import evaluate_retrieval
from utils import log_timing
import pandas as pd
import mlflow


# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("retrieval_and_evaluation")

## setting initial values :
model="multi-qa-mpnet-base-dot-v1"
k = 10   # K for Precision/Recall@k
stype = Metric.COSINE # Similiarity type
with mlflow.start_run(run_name=f"{stype}_Similiarity") as run:

    ret = Retrieval()
    # query_text = input("Enter your query :")
    ground_truth_path = GROUND_TRUTH
    gt_df = pd.read_csv(ground_truth_path)
    gt_df=gt_df[gt_df["split"]=="train"]
    all_results = []

    for idx, row in gt_df.iterrows():
        query_text = row["query"]
        with log_timing(f"Query_Embedding_{idx}"):
            query_embed = ret.query_to_vector(query_text)
        with log_timing(f"Data_Retrieval_{idx}"):
            _, product_results = ret.data_retrieval(query_text, query_embed, type=stype)

        eval_result = evaluate_retrieval(
                product_results,
                k=k,
                run_name=f"retrieval_l2_query_{idx}"
            )
        all_results.append({"query": query_text, "metrics": eval_result})

        mlflow.log_metric(f"K_value", k)
        mlflow.log_metric(f"Precision_K{idx}", eval_result[f"precision_{k}"])
        mlflow.log_metric(f"Recall_K_{idx}", eval_result[f"recall_{k}"])
        print(f"Query {idx}: {query_text}\nMetrics: {eval_result}\n")

    # rec = Recommendation(query_text, generation_results)
    # content = rec.generate_content()
    # print(f"Product Recommendation:\n\n{content}\n")
    # print(f"Explainability:{rec.explainability()}")
