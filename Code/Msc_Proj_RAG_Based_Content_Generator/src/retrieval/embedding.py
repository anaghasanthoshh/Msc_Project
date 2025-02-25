'''
Models to be considered:
1.all-MiniLM-L6-v2
2.multi-qa-MiniLM-L6-cos-v1
3.paraphrase-mpnet-base-v2
'''
# ====================================================================================##
# One-time-run file as it generates embedding for the data.
# ====================================================================================##
#importing  required libraries
import pandas as pd
from io import StringIO
from functools import lru_cache
from data_loader import load_data, PRODUCT_PATH, REVIEW_PATH
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# ====================================================================================##


#load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ====================================================================================##
'''considering that our data is amazon reviews/ product descriptions and are relatively
shorter sentences, we will do basic cleaning like handling of duplicates.
Since sentence transformers automatically handle tokenisation, its not performed'''


# method for cleaning review data

def flatten_dict(data):
    return ";".join([f"{key}:{value}" for key, value in data.items()])


def flatten_list(data):
    return ",".join(f"{i}" for i in data)


def preprocessing_review_data(df):
    #.copy() to avoid updating df by accident
    #df=df.drop_duplicates().copy()
    #to lowercase
    df['title'] = df['title'].fillna('').str.lower()
    df['text'] = df['text'].fillna('').str.lower()
    #convert to just date
    df['timestamp'] = df['timestamp'].dt.date

    return df


# method for cleaning product data
def preprocessing_product_data(df):
    #df=df.drop_duplicates().copy()
    df['title'] = df['title'].fillna('').str.lower()
    df['main_category'] = df['main_category'].fillna('').str.lower()
    df['description'] = df['description'].apply(lambda x: '' if isinstance(x, list) and len(x) == 0 else x)
    df["description"].apply(flatten_list)
    # df["categories"].apply(lambda x: '' if isinstance (x,list) and len(x)==0 else x)
    df["features"] = df["features"].apply(lambda x: '' if isinstance(x, list) and len(x) == 0 else x)
    df["features"] = df["features"].apply(flatten_list)
    df["categories"] = df["categories"].apply(lambda x: '' if isinstance(x, list) and len(x) == 0 else x)
    df["categories"] = df["categories"].apply(flatten_list)
    df["details"] = df["details"].apply(flatten_dict)
    return df


# ====================================================================================##
# Embedding and storing the data in chromadb
# ====================================================================================##

chroma_client = chromadb.Client(settings=chromadb.Settings(
        persist_directory="../../Data/chroma_db"))  # Persistent storage
def chroma_clients_collection():

    #chroma_client.delete_collection(name="product_embeddings")
    product_collection = chroma_client.get_or_create_collection("product_embeddings")
    #chroma_client.delete_collection(name="review_embeddings")
    review_collection = chroma_client.get_or_create_collection("review_embeddings")
    return product_collection, review_collection


# noticed an issue of duplicates being added into collection. Hence, checking for
# existing ids
def fetch_existing_ids(collection):
    # Fetch all existing embeddings from the collection
    existing_embeddings = collection.get()
    existing_ids = set(existing_embeddings['ids'])  # Convert to set for faster lookup
    return existing_ids


def product_embedding(df_product, product_collection):
    existing_ids = fetch_existing_ids(product_collection)
    for _, product in df_product.iterrows():  #unpacked as index,item
        embedding_id = product["parent_asin"]
        if embedding_id not in existing_ids:  #id logic used
            combined_text = f"{product["description"]}{product["title"]}{product["features"]}{product["description"]}{product["details"]}"
            embedding = model.encode(combined_text)

            # Add to ChromaDB
            product_collection.add(
                ids=[product["parent_asin"]],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "title": product["title"],
                    "price": product["price"],
                    "product_details": combined_text
                }])




# define review embedding and adding to chromadb
def review_embedding(df_review, review_collection):
    #existing_ids=fetch_existing_ids(review_collection)

    for _, review in df_review.iterrows():
        #     embedding_id=review["parent_asin"]
        #     if embedding_id not in existing_ids:existing_ids
        combined_text = f"{review["title"]}{review["text"]}"
        embedding = model.encode(combined_text)
        unique_id = str(uuid.uuid4())
        # Add to ChromaDB
        review_collection.add(
            ids=unique_id,
            embeddings=[embedding.tolist()],
            metadatas=[{
                "title": review["title"],
                "text": review["text"],
                "product_id": review["parent_asin"]

            }]

        )



df_product, df_review = load_data(PRODUCT_PATH, REVIEW_PATH)
dfr = preprocessing_review_data(df_review)
dfp = preprocessing_product_data(df_product)
p_collection, r_collection = chroma_clients_collection()
product_embedding(dfp, p_collection)
review_embedding(dfr, r_collection)
