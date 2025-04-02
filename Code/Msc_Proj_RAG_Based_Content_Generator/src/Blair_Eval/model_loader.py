# blair_eval/model_loader.py
from transformers import AutoTokenizer, AutoModel
import torch


def load_model(model_name="hyp1231/blair-roberta-large", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device
