import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(sentences, claims, tokenizer):
    text = list(zip(sentences, claims))
    encoded_dict = tokenizer.batch_encode_plus(
        text, pad_to_max_length=True, return_tensors="pt"
    )
    if encoded_dict["input_ids"].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            text,
            max_length=512,
            pad_to_max_length=True,
            truncation_strategy="only_first",
            return_tensors="pt",
        )
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def get_claim_label_predictions(
    corpus_path, saved_model_path, dataset_path, rationales_path, output_path
):
    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(corpus_path)}
    dataset = jsonlines.open(dataset_path)
    output = jsonlines.open(output_path, "w")
    rationale_selection = jsonlines.open(rationales_path)

    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    config = AutoConfig.from_pretrained(saved_model_path, num_labels=3)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            saved_model_path, config=config
        )
        .eval()
        .to(device)
    )

    LABELS = ["CONTRADICT", "NOT_ENOUGH_INFO", "SUPPORT"]

    with torch.no_grad():
        for data, selection in tqdm(list(zip(dataset, rationale_selection))):
            assert data["id"] == selection["claim_id"]

            claim = data["claim"]
            results = {}
            for doc_id, indices in selection["evidence"].items():
                if not indices:
                    results[doc_id] = {"label": "NOT_ENOUGH_INFO", "confidence": 1}
                else:
                    evidence = " ".join(
                        [corpus[int(doc_id)]["abstract"][i] for i in indices]
                    )
                    encoded_dict = encode([evidence], [claim], tokenizer)
                    label_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[0]
                    label_index = label_scores.argmax().item()
                    label_confidence = label_scores[label_index].item()
                    results[doc_id] = {
                        "label": LABELS[label_index],
                        "confidence": round(label_confidence, 4),
                    }
            output.write({"claim_id": data["id"], "labels": results})

