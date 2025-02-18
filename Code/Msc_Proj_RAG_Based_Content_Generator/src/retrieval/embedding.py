'''
Models to be considered:
1.all-MiniLM-L6-v2
2.multi-qa-MiniLM-L6-cos-v1
3.paraphrase-mpnet-base-v2
'''
# ====================================================================================##
#import required libraries
import pandas as pd
from io import StringIO
from data_loader import load_data
from sentence_transformers import SentenceTransformer
import chromadb
# ====================================================================================##

#load data
df_product=load_data('/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/raw_data/product_sample.json')
df_review=load_data('/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/raw_data/review_sample.json')

#load model
model=SentenceTransformer("all-MiniLM-L6-v2")

# ====================================================================================##
'''considering that our data is amazon reviews/ product descriptions and are relatively
shorter sentences, we will do basic cleaning like handling of duplicates.
Since sentence transformers automatically handle tokenisation, its not performed'''

# method for cleaning review data

def flatten_dict(data):
    return ";".join([f"{key}:{value}" for key,value in data.items()])
def flatten_list(data):
    return ",".join(f"{i}" for i in data)
def preprocessing_review_data(df):
    #.copy() to avoid updating df by accident
    #df=df.drop_duplicates().copy()
    #to lowercase
    df['title']=df['title'].fillna('').str.lower()
    df['text']=df['text'].fillna('').str.lower()
    #convert to just date
    df['timestamp']=df['timestamp'].dt.date

    return df


# method for cleaning product data
def preprocessing_product_data(df):
    #df=df.drop_duplicates().copy()
    df['title']=df['title'].fillna('').str.lower()
    df['main_category']=df['main_category'].fillna('').str.lower()
    df['description']=df['description'].apply(lambda x:'' if isinstance(x,list)and len(x)==0 else x)
    df["description"].apply(flatten_list)
    # df["categories"].apply(lambda x: '' if isinstance (x,list) and len(x)==0 else x)
    df["features"]=df["features"].apply(lambda x: '' if isinstance(x,list) and len(x)==0 else x)
    df["features"]=df["features"].apply(flatten_list)
    df["categories"]=df["categories"].apply(lambda x: '' if isinstance(x,list) and len(x)==0 else x)
    df["categories"]=df["categories"].apply(flatten_list)
    df["details"]=df["details"].apply(flatten_dict)
    return df

# ====================================================================================##
# Embedding and storing the data in chromadb
# ====================================================================================##

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage

def product_embedding(df_product):
    collection = chroma_client.get_or_create_collection("product_embeddings")
    for _,product in df_product.iterrows():#unpacked as index,item
        combined_text=f"{product["description"]}{product["title"]}{product["features"]}{product["description"]}{product["details"]}"
        embedding = model.encode(combined_text)

    # Add to ChromaDB
        collection.add(
            ids= [product["parent_asin"]],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "title": product["title"],
                "price": product["price"],
                "product_details":combined_text
            }]
        )


# define review embedding and adding to chromadb
def review_embedding(df_review):
    collection = chroma_client.get_or_create_collection("review_embeddings")
    for _,review in df_review.iterrows():
        combined_text = f"{review["title"]}{review["text"]}"
        embedding = model.encode(combined_text)

        # Add to ChromaDB
        collection.add(
            ids=[review["parent_asin"]],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "title": review["title"],

                "text":review["text"]

            }]
        )




dfr=preprocessing_review_data(df_review)
dfp=preprocessing_product_data(df_product)
product_embedding(dfp)
review_embedding(dfr)

