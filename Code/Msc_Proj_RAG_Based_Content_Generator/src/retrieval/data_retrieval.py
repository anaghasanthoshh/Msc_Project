import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# sample query
query_text = "Find long-lasting gel nail polish."

chroma_client = chromadb.PersistentClient(path="../../Data/chroma_db")
#print("starting to fetch products")
product_collection=chroma_client.get_collection("product_embeddings")
#print("starting to fetch review")
review_collection=chroma_client.get_collection("review_embeddings")

#Setting how many results need to be retrieved
global k
k=3

# converting the query to embedding
def query_to_vector(query):
    query_embedding = model.encode(query)
    return query_embedding

def data_retrieval(query_embedding):
 # Retrieve from product collection
    product_results = product_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=k
    )

# Retrieve from review collection
    review_results = review_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=k
    )
    product_ids=[item for sublist in product_results['ids'] for item in sublist]

    prod_id_reviews=review_collection.get(
        where={"parent_asin": {"$in": product_ids}}
    )

    return product_results,review_results,prod_id_reviews


if __name__=="__main__":
    query_embed=query_to_vector(query_text)
    product_results, review_results, prod_id_reviews=data_retrieval(query_embed)
    print(f"The product results are: \n{product_results}\n")
    print(f"The review results are: \n{review_results}\n")
    print(f"The reviews for fetched product results are: \n{prod_id_reviews}\n")

##TODO: Remove multiple entries present at the moment in review collection
##Cause : Added new column to collection-parent_asin for reviews and they are entered
## as new data.
##Fix: delete and create again.
'''
Product ids='B00UJOX1Z8', 'B07GV9VGLD', 'B00AGBQK4S'
review fetched=B07GV9VGLD 
We can see that there are some overlap between product and reviews fetched.
'''