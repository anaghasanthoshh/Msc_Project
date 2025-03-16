import chromadb
from sentence_transformers import SentenceTransformer
from embedding import EmbedData

emb = EmbedData()
model = emb.model
# print("starting to fetch products")
product_collection=emb.product_collection

# Setting how many results need to be retrieved
global k
k=5

# converting the query to embedding
def query_to_vector(query):
    query_embedding = model.encode(query)
    print("query encoded")
    return query_embedding

def data_retrieval(query_embedding):
 # Retrieve from product collection
    product_results = product_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=k
    )


    return product_results


if __name__=="__main__":
    query_text=input("Enter your query :")
    query_embed=query_to_vector(query_text)
    product_results=data_retrieval(query_embed)
    print(f"The product results are: \n{product_results}\n")

    #print(f"The reviews for fetched product results are: \n{prod_id_reviews}\n")
