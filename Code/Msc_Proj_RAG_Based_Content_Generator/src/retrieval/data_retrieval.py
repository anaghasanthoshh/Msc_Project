import chromadb
#import embedding as emb
from retrieval.embedding import model
# sample query
query_text = "i want LASHVIEW brand eye lashes "

chroma_client = chromadb.Client(settings=chromadb.Settings(
        persist_directory="../../Data/chroma_db"))
product_collection=chroma_client.get_collection("product_embeddings")
review_collection=chroma_client.get_collection("review_embeddings")


# converting the query to embedding
def query_to_vector(query):
    query_embedding = model.encode(query)
    return query_embedding

def data_retrieval(query_embedding):
 # Retrieve from product collection
    product_results = product_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
    )

# Retrieve from review collection
    review_results = review_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
    )
    return product_results,review_results



query_embed=query_to_vector(query_text)
print(data_retrieval(query_embed))

