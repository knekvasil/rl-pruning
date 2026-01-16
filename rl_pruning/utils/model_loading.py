from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def load_sst2_model():
    # TODO: Modularize model?
    model_name = "textattack/bert-base-uncased-SST-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        dtype=torch.float32,
    )
    return model, tokenizer
