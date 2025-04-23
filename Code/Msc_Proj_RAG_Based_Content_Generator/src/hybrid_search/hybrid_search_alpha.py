# ====================================================================================##
# hybrid search alpha module combining keyword filtering and semantic retrieval
# ====================================================================================##
# importing required libraries
from sklearn.preprocessing import MinMaxScaler
from whoosh.qparser import MultifieldParser,OrGroup
from whoosh.index import open_dir
from sentence_transformers import util
from retrieval.config import WHOOSH_INDEX
from retrieval.embedding import EmbedData
from retrieval.data_retrieval import Retrieval
from sentence_transformers import SentenceTransformer
import numpy as np

# ====================================================================================##
# hybrid search alpha class for keyword, semantic, and reranking
# ====================================================================================##
class HybridSearchAlpha:
    # ====================================================================================##
    # initialize with query text, embedding model, and distance metric
    # ====================================================================================##
    def __init__(self,query_text,model,type="L2"):
        self.ret=Retrieval(model)
        self.model=self.ret.model
        self.merged_data={}
        self.query_text=query_text
        self.product_coll_l2=self.ret.product_coll_l2
        self.product_coll_ip=self.ret.product_coll_ip
        self.product_coll_cosine=self.ret.product_coll_cosine
        self.type=type
    # ====================================================================================##
    # keyword_filter: run whoosh-based keyword search and return top_n items
    # ====================================================================================##
    def keyword_filter(self,index_dir=WHOOSH_INDEX, top_n=5):
        ix = open_dir(index_dir)
        temp_data={}
        filtered_data ={}
        scores=[]
        with ix.searcher() as searcher:
            parser = MultifieldParser([ "metadata"], schema=ix.schema,group=OrGroup.factory(0.9))
            query = parser.parse(self.query_text)
            results = searcher.search(query, limit=top_n)
            for r in results:
                item_id = r["item_id"]
                metadata = r["metadata"]
                score = r.score

                temp_data[item_id] = {
                    "metadata": metadata,
                    "score": score
                }
                scores.append(score)
                # Normalize scores
            if scores:
                scaler = MinMaxScaler()
                norm_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()

                # Second pass: add normalized score to final dict
                for i, (item_id, data) in enumerate(temp_data.items()):
                    filtered_data[item_id] = {
                        "metadata": data["metadata"],
                        "score": norm_scores[i]
                        }

        return filtered_data
    # ====================================================================================##
    # semantic_search: encode query and fetch top_k semantic matches
    # ====================================================================================##
    def semantic_search(self,query_text,k=5):
        semantic_data={}
        query_embedding=self.model.encode(query_text)

        if self.type=="L2":

            product_results = self.product_coll_l2.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k

                )
        elif self.type=="COSINE":
            coll_size=self.product_coll_cosine.count()
            product_results = self.product_coll_cosine.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k

            )
        elif self.type == "IP":
            product_results = self.product_coll_ip.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k

            )
        for i in range(len(product_results['ids'][0])):
            item_id = product_results['ids'][0][i]
            metadata = product_results['metadatas'][0][i]['metadata']
            distance = product_results['distances'][0][i]

            if self.type == "COSINE":
                score = 1 - distance  # Convert cosine distance → similarity
            elif self.type == "L2":
                score = 1 / (1 + distance)  # Lower distance = higher similarity
            elif self.type == "IP":
                score = distance  # Already higher = better
            else:
                score = 0  # fallback


            semantic_data[item_id] = {
                    "metadata": metadata,
                    "score": score
                }
        return semantic_data
    # ====================================================================================##
    # combine_results: merge keyword and semantic outputs, checking for conflicts
    # ====================================================================================##
    def combine_rerank_results(self, filtered_data, semantic_data):
        self.merged_data = {}

        # Merge both dicts
        for item_id, data in {**semantic_data, **filtered_data}.items():
            sem_score = semantic_data.get(item_id, {}).get("score", 0.0)
            lex_score = filtered_data.get(item_id, {}).get("score", 0.0)

            self.merged_data[item_id] = {
                "metadata": data["metadata"],
                "semantic_score": sem_score,
                "keyword_score": lex_score
            }

        return self.merged_data
    # ====================================================================================##
    # rerank_by_alpha_score: compute  similarity scores and sort
    # ====================================================================================##
    def alpha_rerank(self, alpha=0.5):
        reranked = []
        for item_id, data in self.merged_data.items():
            final_score = alpha * data["semantic_score"] + (1 - alpha) * data["keyword_score"]
            reranked.append((item_id, data["metadata"], final_score))

        sorted_results = sorted(reranked, key=lambda x: x[2], reverse=True)
        eval_data={self.query_text:[item[0] for item in sorted_results]}
        return eval_data  # return ordered item_ids


# ====================================================================================##
# command-line interface for hybrid search alpha demonstration
# ====================================================================================##

if __name__=="__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text="black porcelain tile"
    hybrid=HybridSearchAlpha(query_text,model,"COSINE")
    filtered_data=hybrid.keyword_filter()
    semantic_data=hybrid.semantic_search(query_text)
    combine_rerank_results=hybrid.combine_rerank_results(filtered_data,semantic_data)
    print(f"Filtered data:{filtered_data}\n\n")
    print(f"Semantic data:{semantic_data}")
    print(combine_rerank_results)
    alpha_rerank=hybrid.alpha_rerank()








