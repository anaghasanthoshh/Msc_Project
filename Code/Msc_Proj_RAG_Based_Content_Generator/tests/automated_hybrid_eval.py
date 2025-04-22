from hybrid_search.hybrid_search import HybridSearch
from retrieval.config import GROUND_TRUTH
from sentence_transformers import SentenceTransformer
from retrieval.data_retrieval import Metric
from retrieval.retrieval_eval import evaluate_retrieval
from utils.utils import log_timing
import pandas as pd
import mlflow

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Test_Hybrid_retrieval_evaluation")

# setting initial values :
#model_name="all-MiniLM-L6-v2"
model_name="multi-qa-mpnet-base-dot-v1"
model=SentenceTransformer(model_name)
k_list = [3,5,10]   # K for Precision/Recall@k
type_List=[Metric.L2.value,Metric.COSINE.value,Metric.IP.value]
ground_truth_path = GROUND_TRUTH
gt_df = pd.read_csv(ground_truth_path)
gt_df=gt_df[gt_df["split"]=="test"]
all_results = []
all_precisions = []
all_recalls = []
all_mrr =[]
metrics_artifact={}
for type in type_List:
    for k in k_list:
        with (mlflow.start_run(run_name=f"Model:{model_name}_K:{k}_Similiarity:{type}") as run):
            for idx, row in gt_df.iterrows():
                query_text = row["query"]
                with log_timing(f"Hybrid_Data_Retrieval_{idx}"):
                    hybrid = HybridSearch(query_text, model)
                    filtered_data = hybrid.keyword_filter()
                    query_embedding, semantic_data = hybrid.semantic_search(query_text)
                    merged_data = hybrid.combine_rerank_results(filtered_data, semantic_data)
                    final_retrieved_data = hybrid.cosine_similiarity_rerank(query_embedding)
                with log_timing(f"Hybrid_Retrieval_Evaluation_{idx}"):
                    eval_result = evaluate_retrieval(
                        final_retrieved_data,
                        k=k,
                        run_name=f"evaluation_query_{idx}"
                    )
                all_results.append({"query": query_text, "metrics": eval_result})
                all_precisions.append(eval_result[f"precision_{k}"])
                all_recalls.append(eval_result[f"recall_{k}"])
                all_mrr.append(eval_result["MRR_score"])

                mlflow.log_param(f"K_value", k)
                mlflow.log_param("model",model_name)
                mlflow.log_metric(f"Precision_K{idx}", eval_result[f"precision_{k}"])
                mlflow.log_metric(f"Recall_K_{idx}", eval_result[f"recall_{k}"])
                mlflow.log_metric("MRR_score", eval_result["MRR_score"])

                print(f"Query {idx}: {query_text}\nMetrics: {eval_result}\n")

            avg_precision = (sum(all_precisions)/len(all_precisions)).iloc[0]
            avg_recall = (sum(all_recalls)/len(all_recalls)).iloc[0]
            avg_mrr = (sum(all_mrr)/len(all_mrr)).iloc[0]
            mlflow.log_metric("Average_Precision",avg_precision)
            mlflow.log_metric("Average_Recall",avg_recall)
            mlflow.log_metric("Average_MRR_score",avg_mrr)
            print(f"Average recall: {avg_recall.item}")
            print(f"Average Precision: {avg_precision}")
            print(f"Average MRR :{avg_mrr}")
