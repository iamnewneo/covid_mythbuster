import warnings
import torch
from transformers import logging
from covid_mythbuster.inference.covid import (
    AbstractRetriever,
    RationaleSelector,
    LabelPredictor,
)
from covid_mythbuster.config import RATIONALE_MODEL_PATH, LABEL_MODEL_PATH

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

rationale_selection_model = RATIONALE_MODEL_PATH
label_prediction_model = LABEL_MODEL_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(claim):
    abstract_retriever = AbstractRetriever()

    rationale_selector = RationaleSelector(
        model=rationale_selection_model,
        selection_method="topk",
        threshold=0.5,
        device=device,
    )
    label_predictor = LabelPredictor(
        model=label_prediction_model, keep_nei=True, threshold=0.5, device=device
    )

    abstracts = abstract_retriever(claim=claim)

    print(f"Claim: {claim}")
    print("Top 10: List of retreived abstracts")
    for abstract in abstracts:
        abstract_text = " ".join(abstract["abstract"])
        print(f"Title: {abstract['title']}")
        print(f"Abstract: {abstract_text}")
        print("\n\n")

    rationales = rationale_selector(claim=claim, documents=abstracts)
    predicted_labels = label_predictor(claim=claim, retrievals=rationales)

    print(f"Claim: {claim}")
    for doc_label in predicted_labels:
        if doc_label["label"] != "NOT_ENOUGH_INFO":
            sentences = [doc_label["abstract"][i] for i in doc_label["evidence"]]
            sentences = " ".join(sentences)
            print(f"Label: {doc_label['label']}")
            print(f"Evidence Sentence: {sentences}")


if __name__ == "__main__":
    # claim = "Mass masking reduces COVID-19 transmission rates."
    claim = "Masks do not help with COVID-19."
    main(claim=claim)
