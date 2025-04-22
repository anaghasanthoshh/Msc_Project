import requests

class Recommendation:
    def __init__(self,query,result_context):
        self.query=query
        self.result_context=result_context
        self.prompt=''
        self.model="llama2"
        self.output=''



    def generate_content(self):
        # Defining a prompt
        full_prompt = (
    "You are a helpful and knowledgeable assistant for product recommendations.\n"
    "The user has asked a question, and you've been given a list of retrieved products in the format:\n"
    "product_id: product description.\n\n"
    "DO NOT MAKE UP INFORMATION.ONLY RECOMMEND WITH AVAILABLE DETAILS.IF DATA IS NOT PROVIDED,ACCEPT THAT WE DONT HAVE SUFFICIENT"
    ""
    "Your job is to read through the product information and recommend the items.\n"
    "Present them as a friendly list with brief explanations for each product, showing how they relate to the user's query.\n\n"
    "Only recommend from the provided context. Do not make up information.Include the ID of the products\n"
    "Begin your response with: 'Based on your query, here are some products I recommend:'\n\n"
    f"Context:\n{self.result_context}\n\n"
    f"User Query:\n{self.query}\n\n"
    "Answer:"
                    )

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model, # the model that we pulled using ollama
                "prompt": full_prompt,
                "stream": False  # set to False to get the full response at once
            }
        )

        if response.status_code == 200:
            self.output = response.json()["response"]
            return self.output.strip()
        else:
            print("Error communicating with Ollama:", response.text)
            return None

    def explainability(self):
        explanation_prompt = explanation_prompt = (
    "You are an assistant that explains why specific recommendations were made based on exact context and matching features.\n\n"
    f"Context:\n{self.result_context}\n\n"
    f"User Query:\n{self.query}\n\n"
    f"Recommendations:\n{self.output.strip()}\n\n"
    "For each recommended item:\n"
    "- Identify the **exact words or features** that matched the query.\n"
    "- Refer to **specific phrases from the product description** as proof.\n"
    "- Include the **item ID**.\n"
    "- Explain briefly **why this item was selected**, using only the context provided.\n\n"
    " Do NOT invent or assume anything. Only use the data in the context.\n"
    "Keep the explanations factual, concise, and easy to understand."
)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,  # the model that we pulled using ollama
                "prompt":explanation_prompt,
                "stream": False  # set as False,to get the full response at once
            }
        )
        if response.status_code == 200:
            self.output = response.json()["response"]
            return self.output.strip()
        else:
            print("Error communicating with Ollama:", response.text)
            return None






