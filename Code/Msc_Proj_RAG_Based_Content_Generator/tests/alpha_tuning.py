from hybrid_search.hybrid_search_alpha import HybridSearchAlpha
from retrieval.config import GROUND_TRUTH
from sentence_transformers import SentenceTransformer
from retrieval.data_retrieval import Metric
from retrieval.retrieval_eval import evaluate_retrieval
from utils.utils import log_timing
import pandas as pd
import mlflow
import numpy as np

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Alpha tuning")

# setting initial values :
model_name="all-MiniLM-L6-v2"
#model_name="multi-qa-mpnet-base-dot-v1"
model=SentenceTransformer(model_name)

k_list=[5]   # K for Precision/Recall@k
type=Metric.COSINE.value

alphas = np.arange(0.0, 1.05, 0.05)
ground_truth_path = GROUND_TRUTH
gt_df = pd.read_csv(ground_truth_path)
gt_df=gt_df[gt_df["split"]=="train"]
all_results = []
all_precisions = []
all_recalls = []
all_mrr =[]
metrics_artifact={}

for alpha in alphas:
    for k in k_list:
        with (mlflow.start_run(run_name=f"Model:{model_name}_K:{k}_alpha:{alpha}") as run):
            for idx, row in gt_df.iterrows():
                query_text = row["query"]
                hybrid = HybridSearchAlpha(query_text, model, type)
                filtered_data = hybrid.keyword_filter()
                semantic_data = hybrid.semantic_search(query_text)
                combine_rerank_results = hybrid.combine_rerank_results(filtered_data, semantic_data)
                alpha_rerank_results=hybrid.alpha_rerank(alpha)
                eval_result = evaluate_retrieval(
                        alpha_rerank_results,
                        k=k,
                        run_name=f"evaluation_query"
                    )

                all_precisions.append(eval_result[f"precision_{k}"])
                all_recalls.append(eval_result[f"recall_{k}"])
                all_mrr.append(eval_result["MRR_score"])

                mlflow.log_param(f"K_value", k)
                mlflow.log_param("model",model_name)
                mlflow.log_param("alpha",alpha)

            avg_precision = (sum(all_precisions) / len(all_precisions)).iloc[0]
            avg_recall = (sum(all_recalls) / len(all_recalls)).iloc[0]
            avg_mrr = (sum(all_mrr) / len(all_mrr)).iloc[0]
            all_results.append((alpha,avg_precision,avg_recall,avg_mrr))

            mlflow.log_metric("Average_Precision",avg_precision)
            mlflow.log_metric("Average_Recall",avg_recall)
            mlflow.log_metric("Average_MRR_score",avg_mrr)

best_alpha_mrr = max(all_results, key=lambda x: x[3])
best_alpha_precision=max(all_results, key=lambda x: x[1])
print(f"Best alpha (MRR): {best_alpha_mrr[0]} with MRR = {best_alpha_mrr[3]}")
print(f"Best alpha (Precision): {best_alpha_precision[0]} with Precision = {best_alpha_precision[1]}")
