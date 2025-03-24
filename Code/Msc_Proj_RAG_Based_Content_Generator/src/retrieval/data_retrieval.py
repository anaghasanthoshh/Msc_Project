from retrieval.embedding import EmbedData

class Retrieval:

    def __init__(self,k=5):
        self.emb = EmbedData()
        self.model = self.emb.model
        # print("starting to fetch products")
        self.product_collection=self.emb.product_collection
        self.k=k

    # converting the query to embedding
    def query_to_vector(self,query):
        query_embedding = self.model.encode(query)
        #print("query encoded")
        return query_embedding

    def data_retrieval(self,query_text,query_embedding):
    # Retrieve from product collection
        product_results = self.product_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=self.k
        )
        #print(product_results)
        prod_results=product_results['ids'][0]
        gen_result={}
        for i in range(len(product_results['ids'][0])):
            gen_result[product_results['ids'][0][i]]=product_results['metadatas'][0][i]['metadata']

        eval_results={query_text:prod_results}

        return gen_result,eval_results


if __name__=="__main__":
    ret=Retrieval()
    query_text=input("Enter your query :")
    query_embed=ret.query_to_vector(query_text)
    _,product_results=ret.data_retrieval(query_text,query_embed)
    print(f"The product results are: \n{product_results}\n")

    #print(f"The reviews for fetched product results are: \n{prod_id_reviews}\n")
