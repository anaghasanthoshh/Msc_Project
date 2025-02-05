import pandas as pd
import chromadb
import json
import ollama
from glob import glob
from PyPDF2 import PdfReader
from tqdm import tqdm


#=====================display settings for dataframe==================================
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

#=====================Trying out a product based query on general LLM===================
query = "I need a shampoo that prevents hair fall but doesn’t dry my scalp."
query_model = "llama3"

def get_ollama_response(question, model):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ], options={"seed": 42}
    )

    return response["message"]["content"]

response = get_ollama_response(query, query_model)
print(response)

#run