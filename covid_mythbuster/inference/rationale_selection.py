import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def claim_rationale_selection(
    corpus_path, saved_model_path, dataset_path, retrieved_abstracts_path, output_path
):
    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(corpus_path)}
    dataset = jsonlines.open(dataset_path)
    abstract_retrieval = jsonlines.open(retrieved_abstracts_path)

    threshold = 0.5
    only_rationale = True
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = (
        AutoModelForSequenceClassification.from_pretrained(saved_model_path)
        .to(device)
        .eval()
    )

    results = []

    with torch.no_grad():
        for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
            assert data["id"] == retrieval["claim_id"]
            claim = data["claim"]

            evidence_scores = {}
            for doc_id in retrieval["doc_ids"]:
                doc = corpus[doc_id]
                sentences = doc["abstract"]
                batch = zip(sentences, [claim] * len(sentences))

                encoded_dict = tokenizer.batch_encode_plus(
                    list(batch) if only_rationale else sentences,
                    pad_to_max_length=True,
                    return_tensors="pt",
                )
                encoded_dict = {
                    key: tensor.to(device) for key, tensor in encoded_dict.items()
                }
                sentence_scores = (
                    torch.softmax(model(**encoded_dict)[0], dim=1)[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )
                evidence_scores[doc_id] = sentence_scores
            results.append(
                {"claim_id": retrieval["claim_id"], "evidence_scores": evidence_scores}
            )

    output = jsonlines.open(output_path, "w")

    for result in results:
        evidence = {
            doc_id: (sentence_scores >= threshold).nonzero()[0].tolist()
            for doc_id, sentence_scores in result["evidence_scores"].items()
        }
        output.write({"claim_id": result["claim_id"], "evidence": evidence})

