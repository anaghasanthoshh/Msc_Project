from whoosh.qparser import MultifieldParser,OrGroup
from whoosh.index import open_dir
from sentence_transformers import util
from retrieval.config import WHOOSH_INDEX
from retrieval.embedding import EmbedData
from retrieval.data_retrieval import Retrieval
from sentence_transformers import SentenceTransformer

class HybridSearch:

    def __init__(self,query_text,model):
        self.ret=Retrieval(model)
        self.model=self.ret.model
        self.merged_data={}
        self.query_text=query_text

    def keyword_filter(self,index_dir=WHOOSH_INDEX, top_n=10):
        ix = open_dir(index_dir)
        filtered_data ={}
        with ix.searcher() as searcher:
            parser = MultifieldParser([ "metadata"], schema=ix.schema,group=OrGroup.factory(0.9))
            query = parser.parse(self.query_text)
            results = searcher.search(query, limit=top_n)
            for r in results:
            # filtered_ids = [r["item_id"] for r in results]
            # data=[r["metadata"] for r in results]
                filtered_data[r["item_id"]]=r["metadata"]



        return filtered_data

    def semantic_search(self,query_text,k=10):
        semantic_data={}
        query_embedding=self.model.encode(query_text,convert_to_tensor=True)
        product_results = self.ret.product_coll_l2.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k

        )
        for i in range(len(product_results['ids'][0])):
            semantic_data[product_results['ids'][0][i]] = product_results['metadatas'][0][i]['metadata']
        print(semantic_data)
        return query_embedding,semantic_data

    def combine_rerank_results(self,filtered_data,semantic_data):
        for k, v in semantic_data.items():
            if k in filtered_data and filtered_data[k] != v:
                raise ValueError(f"Conflict at key '{k}': {filtered_data[k]} != {v}")
        self.merged_data={**semantic_data, **filtered_data}
        return self.merged_data

    def cosine_similiarity_rerank(self,query_embedding):
        scored_data=[]
        for key,value in self.merged_data.items():
            product_text = value

            product_embed = self.model.encode(product_text,convert_to_tensor=True)
            score= util.cos_sim(query_embedding, product_embed)[0][0].item()
            scored_data.append((key,product_text,score))


        # Rerank
        final_results = sorted(scored_data, key=lambda x: x[2], reverse=True)
        return final_results


if __name__=="__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text="black porcelain tile"
    hybrid=HybridSearch(query_text,model)
    filtered_data=hybrid.keyword_filter()
    query_embedding,semantic_data=hybrid.semantic_search(query_text)
    merged_data=hybrid.combine_rerank_results(filtered_data,semantic_data)
    sorted_data=hybrid.cosine_similiarity_rerank(query_embedding)







