from os import cpu_count
from covid_mythbuster.inference import (
    write_retrieved_abstracts,
    claim_rationale_selection,
    evaluate_rationale_predictions,
    get_claim_label_predictions,
    evaluate_label_predictions,
)


def main():
    dataset_path = "./data/claims_dev.jsonl"
    abstract_output_file_path = "./predictions/abstract_retrieved.jsonl"
    corpus_path = "./data/corpus.jsonl"
    rationale_saved_model_path = "./models/rationale_roberta_large_scifact/"
    label_saved_model_path = "./models/label_roberta_large_scifact/"

    rationale_output_path = "./predictions/rationale_selected.jsonl"
    label_output_path = "./predictions/label_selected.jsonl"
    print("Running abstarct")
    write_retrieved_abstracts(
        dataset_path=dataset_path, output_file_path=abstract_output_file_path
    )

    # claim_rationale_selection(
    #     corpus_path=corpus_path,
    #     saved_model_path=rationale_saved_model_path,
    #     dataset_path=dataset_path,
    #     retrieved_abstracts_path=abstract_output_file_path,
    #     output_path=rationale_output_path,
    # )

    evaluate_rationale_predictions(
        dataset_path=dataset_path, rationale_path=rationale_output_path,
    )

    print("Running Label Predictions")
    get_claim_label_predictions(
        corpus_path=corpus_path,
        saved_model_path=label_saved_model_path,
        dataset_path=dataset_path,
        rationales_path=rationale_output_path,
        output_path=label_output_path,
    )
    evaluate_label_predictions(
        corpus_path=corpus_path,
        dataset_path=dataset_path,
        label_pred_path=label_output_path,
    )


if __name__ == "__main__":
    main()
