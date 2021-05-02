from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

paraphrase = tokenizer.encode_plus(
    sequence_0,
    sequence_2,
    add_special_tokens=True,
    max_length=50,
    return_token_type_ids=True,
    truncation=True,
    padding="max_length",
    return_attention_mask=True,
    return_tensors="pt",
)
print(paraphrase["input_ids"])
print(paraphrase["attention_mask"])
