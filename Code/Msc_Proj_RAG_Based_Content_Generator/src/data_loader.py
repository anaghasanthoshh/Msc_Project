import pandas as pd
import os
import sys
from io import StringIO
from postgres_connection import get_db_connection
from dotenv import load_dotenv


PRODUCT_PATH='/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/raw_data/product_sample.json'
REVIEW_PATH='/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/raw_data/review_sample.json'


#load the data  and return metadata,dataframes for tracking details in MLflow_tracking.py
def load_data(p_path,r_path):
    product_df=pd.read_json(p_path)
    p_metadata={
        "dat_path":p_path,
        "row_count":product_df.shape[0],
        "column_names":product_df.columns
        }
    review_df=pd.read_json(r_path)
    r_metadata={
        "dat_path":r_path,
        "row_count":review_df.shape[0],
        "column_names":review_df.columns
        }


    #return metadata
    return product_df,review_df


# method for cleaning review data

def flatten_dict(data):
    return ";".join([f"{key}:{value}" for key, value in data.items()])


def flatten_list(data):
    return ",".join(f"{i}" for i in data)
##TODO: Need check for language,removal of html tags from text
def preprocessing_review_data(df):
    #.copy() to avoid updating df by accident
    #df=df.drop_duplicates().copy()
    #to lowercase
    df['title'] = df['title'].fillna('').str.lower()
    df['text'] = df['text'].fillna('').str.lower()
    #convert to just date
    df['rating']=df['rating'].fillna(3.0)
    df['timestamp'] = df['timestamp'].dt.date
    df['helpful_vote']=df['helpful_vote'].fillna(0)
    file_path='../Data/processed_data/processed_review_data.json'
    if os.path.exists(file_path):
        os.remove(file_path)
    df.to_csv(file_path,sep='^')

    return df


# method for cleaning product data,used by both sql and vector embedding
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
    df["average_rating"]=df["average_rating"].fillna(3.0)#default vslue
    df["rating_number"]=df["rating_number"].fillna(0)
    df["store"] = df["store"].replace({"NULL": "Unknown", None: "Unknown", "": "Unknown"}).fillna("Unknown").astype(str)
    df["price"]=df["price"].fillna(000.000)
    file_path='../Data/processed_data/processed_prod_data.json'
    if os.path.exists(file_path):
        os.remove(file_path)
    df.to_csv(file_path, sep='^')

    return df


def load_data_to_postgres(p_df,r_df):
    # selecting required columns for sql db from product_df
    p_table_name='product_description'
    p_selected_columns=['title','average_rating','rating_number','price','store','main_category','parent_asin']
    p_data=p_df[p_selected_columns]

    # selecting required columns for sql db from review_df
    r_table_name='product_review'
    r_selected_columns=['parent_asin','title','rating','helpful_vote','verified_purchase']
    r_data=r_df[r_selected_columns]

# initiating db connection
    load_dotenv()
    conn=get_db_connection()
    cursor=conn.cursor()
    # point to custom schema, and not public
    schema = os.getenv("DATABASE_SCHEMA", "public")
    cursor.execute(f"SET search_path TO {schema};")
# defining buffer to store data
    p_csv_buffer = StringIO()
    r_csv_buffer=StringIO()

# convert df to csv in memory
    p_data.to_csv(p_csv_buffer,sep="^", index=False, header=False)
    #to point to the initial position
    p_csv_buffer.seek(0)

    r_data.to_csv(r_csv_buffer,sep="^",index=False, header=False)
    r_csv_buffer.seek(0)


    try:
        cursor.copy_from(p_csv_buffer,p_table_name, sep="^", null="NULL", columns=p_data.columns)

        print("inserted {len(p_data)} rows into {p_table_name}")

        cursor.copy_from(r_csv_buffer,r_table_name, sep="^", null="NULL", columns=r_data.columns)
        print("inserted {len(p_data)} rows into {p_table_name}")

        conn.commit()
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

## calling the functions performing the data_cleaning/data loading
product_df,review_df=load_data(PRODUCT_PATH,REVIEW_PATH)
p_df=preprocessing_product_data(product_df)
r_df=preprocessing_review_data(review_df)
print(product_df["description"])
#load_data_to_postgres(p_df,r_df)