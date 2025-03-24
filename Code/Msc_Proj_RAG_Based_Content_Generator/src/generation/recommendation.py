import requests

class Recommendation:
    def __init__(self,query,result_context):
        self.query=query
        self.result_context=result_context
        self.prompt=''
        self.model="llama2"
        self.output=''

    import requests

    def generate_content(self):
        # Defining a prompt
        full_prompt = (f"You are a helpful assistant."
                       f"Answer the user query with the products supplied consisting of product id:product description pair in below format"
                       f"Based on your query, here are the list of items I would recommend."
                       f"itemid:small description of the item based on context\n"
                       f"\n\nContext:{self.result_context}\n\nQuestion: {self.query}\n\nAnswer:")

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
        explanation_prompt = (
            f"You are an assistant that explains recommendations to users.\n"
            f"Context: {self.result_context}\n\n"
            f"User query: {self.query}\n"
            f"Recommendations given: {self.output.strip()}\n\n"
            f"Explain in simple terms which word match or feature led to the recommendation , based on the context."
            f"Include the exact item id and the few words from product description as proof."
            f"Do not make up stuff.Strictly include data only from the result_context ")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,  # the model that we pulled using ollama
                "prompt":explanation_prompt,
                "stream": False  # set to False to get the full response at once
            }
        )
        if response.status_code == 200:
            self.output = response.json()["response"]
            return self.output.strip()
        else:
            print("Error communicating with Ollama:", response.text)
            return None






