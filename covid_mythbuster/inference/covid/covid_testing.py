import time
import torch
import pandas as pd
import warnings
from tqdm import tqdm
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


def get_empty_row():
    temp_row = {
        "id": None,
        "actual_label": None,
        "pred_label": None,
        "confidence": None,
        "abstract_evidence": None,
    }
    return temp_row


def main():
    df = pd.read_csv("./data/covid_fakenews_data.csv")
    df = df.sample(frac=1).reset_index(drop=True)

    df_out = pd.DataFrame(
        columns=[
            "id",
            "claim",
            "actual_label",
            "pred_label",
            "confidence",
            "abstract_evidence",
        ],
        dtype=object,
    )
    df_errors = pd.DataFrame(columns=["error"], dtype=object)
    total_rows = len(df)
    for idx, row in tqdm(df.iterrows(), total=total_rows):
        try:
            claim = row["claim"]
            abstracts = abstract_retriever(claim=claim, k=20)

            if len(abstracts) > 0:
                rationales = rationale_selector(claim=claim, documents=abstracts)
                predicted_labels = label_predictor(claim=claim, retrievals=rationales)
                for pred_label in predicted_labels:
                    sentences = [
                        pred_label["abstract"][i] for i in pred_label["evidence"]
                    ]
                    sentences = " ".join(sentences)
                    label_confidence = pred_label["label_confidence"]
                    label = pred_label["label"]

                    temp_row = get_empty_row()
                    temp_row["id"] = row["id"]
                    temp_row["claim"] = row["claim"]
                    temp_row["actual_label"] = row["label"]
                    temp_row["pred_label"] = label
                    temp_row["confidence"] = label_confidence
                    temp_row["abstract_evidence"] = sentences

                    df_out = df_out.append(temp_row, ignore_index=True)
                    df_out.to_csv("./predictions/covid_fakenews_out.csv", index=False)
        except Exception as e:
            error_string = str(row["id"]) + " " + str(e)
            df_errors.append({"error": error_string}, ignore_index=True)
            df_errors.to_csv("./predictions/covid_errors.csv", index=False)

        time.sleep(3)


if __name__ == "__main__":
    main()
