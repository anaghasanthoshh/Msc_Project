from retrieval.embedding import EmbedData
from enum import Enum
from sentence_transformers import SentenceTransformer



class Metric(Enum):
    COSINE = "COSINE"
    IP = "IP"
    L2 = "L2"


class Retrieval:

    def __init__(self,model, k=5):
        self.emb = EmbedData(model)
        self.model = model
        # print("starting to fetch products")
        self.product_coll_l2 = self.emb.product_coll_l2
        self.product_coll_cosine = self.emb.product_coll_cosine
        self.product_coll_ip = self.emb.product_coll_ip
        self.k = k

    # converting the query to embedding
    def query_to_vector(self, query):
        query_embedding = self.model.encode(query)
        return query_embedding

    def data_retrieval(self, query_text, query_embedding, type="L2"):
        # Retrieve from product collection
        global product_results
        if type == Metric.L2:
            #print("l2 collection count:", len(self.product_coll_l2.get()['ids']))
            product_results = self.product_coll_l2.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.k,

            )
        elif type == Metric.COSINE:
            #print("Cosine collection count:", len(self.product_coll_cosine.get()['ids']))
            product_results = self.product_coll_cosine.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.k
            )
        elif type == Metric.IP:
            #print("ip collection count:", len(self.product_coll_ip.get()['ids']))
            product_results = self.product_coll_ip.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.k
            )


        #print(product_results)
        prod_results = product_results['ids'][0]
        #print(prod_results)
        gen_result = {}
        for i in range(len(product_results['ids'][0])):
            gen_result[product_results['ids'][0][i]] = product_results['metadatas'][0][i]['metadata']

        eval_results = {query_text: prod_results}

        return gen_result, eval_results






if __name__ == "__main__":
    model=SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    ret = Retrieval(model,10)
    query_text = input("Enter your query :")
    query_embed = ret.query_to_vector(query_text)
    gen_results, product_results = ret.data_retrieval(query_text, query_embed,Metric.L2)
    print(f"The product results are: \n{product_results}\n")
    print(gen_results)

    #print(f"The reviews for fetched product results are: \n{prod_id_reviews}\n")
