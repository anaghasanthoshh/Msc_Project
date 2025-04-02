from Blair_Eval.model_embedding import get_embedding
import torch
from tqdm import tqdm


def build_retrieved_results(queries_df, product_id_to_title, tokenizer, model, device, top_k=10):
    """
    For each query, retrieve top_k product IDs from the GLOBAL candidate pool.

    Returns:
        dict: { query: [retrieved product IDs] }
    """
    retrieved_results = {}

    global_candidate_ids = list(product_id_to_title.keys())  # The entire pool

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
        query_text = row["query"]
        ground_truth_id = row["ground_truth_id"]

        # Optional skip if query not in GT dict
        if ground_truth_id not in product_id_to_title:
            continue

        query_emb = get_embedding(query_text, tokenizer, model, device)

        scores = []
        for pid in global_candidate_ids:
            title = product_id_to_title.get(pid, "")
            if not title:
                continue
            product_emb = get_embedding(title, tokenizer, model, device)
            score = torch.nn.functional.cosine_similarity(query_emb, product_emb).item()
            scores.append((pid, score))

        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        ranked_ids = [pid for pid, _ in ranked[:top_k]]

        retrieved_results[query_text] = ranked_ids

    return retrieved_results