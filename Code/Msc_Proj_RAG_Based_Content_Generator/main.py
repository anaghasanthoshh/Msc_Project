from retrieval.data_retrieval import Retrieval,Metric
from retrieval.retrieval_eval import evaluate_retrieval
from generation.recommendation import Recommendation
from sentence_transformers import SentenceTransformer
from utils.utils import log_timing,print_banner
from tabulate import tabulate
import mlflow

## setting initial values :

#model_name="all-MiniLM-L6-v2"
model_name="multi-qa-mpnet-base-dot-v1"
model=SentenceTransformer(model_name)
kr=10
k=5

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("retrieval_and_evaluation")
with mlflow.start_run(run_name="baseline_run") as run:
    ret = Retrieval(model=model,k=kr)

## Block for executing single query ###
    query_text = input("\nEnter your query :\n")
    while(not query_text):
        print("No query provided. Cannot proceed.")
        query_text = input("\nEnter your query :\n")

    with log_timing("Query Embedding"):
        query_embed = ret.query_to_vector(query_text)
    with log_timing("Data Retrieval"):
        generation_input, product_results = ret.data_retrieval(query_text, query_embed, Metric.L2)
    print_banner(
        f"Fetching Data :\n",
              f"Top {kr} product IDs fetched : {product_results}"

    )

    product_results = {query_text: product_results[query_text][:k]}
    evaluation_results = evaluate_retrieval(product_results, k=5, run_name="retrieval_l2")
    if evaluation_results is None:
       print("Query not  in ground truth document.\n"
             "Skipping evaluation and proceeding to generation")
    else:
        print_banner("Retrieval Evaluation Metrics :\n",f"{tabulate(evaluation_results,headers='keys', tablefmt='github')}\n")


    with log_timing("Data Generation"):
      rec = Recommendation(query_text, generation_input)
      content = rec.generate_content()
      print_banner(f"Product Recommendation:\n{content}\n",)
      #print(f"Product Recommendation:\n\n{content}\n")
      print_banner(f"Explainability :\n{rec.explainability()}")




