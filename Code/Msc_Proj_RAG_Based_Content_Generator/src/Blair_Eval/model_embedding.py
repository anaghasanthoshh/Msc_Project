# blair_eval/embedding_utils.py
import torch

def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()  # CLS token