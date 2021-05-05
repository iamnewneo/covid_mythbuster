import jsonlines
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def safe_divide(num, denom):
    if denom == 0:
        return 0
    else:
        return num / denom


def compute_f1_score(counts, difficulty=None):
    correct_key = "correct" if difficulty is None else f"correct_{difficulty}"
    precision = safe_divide(counts[correct_key], counts["retrieved"])
    recall = safe_divide(counts[correct_key], counts["relevant"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def is_correct(pred_sentence, pred_sentences, gold_sets):
    """
    A predicted sentence is correctly identified if it is part of a gold
    rationale, and all other sentences in the gold rationale are also
    predicted rationale sentences.
    """
    for gold_set in gold_sets:
        gold_sents = gold_set["sentences"]
        if pred_sentence in gold_sents:
            if all([x in pred_sentences for x in gold_sents]):
                return True
            else:
                return False

    return False


def evaluate_rationale_predictions(dataset_path, rationale_path):
    dataset = jsonlines.open(dataset_path)
    rationale_selection = jsonlines.open(rationale_path)

    counts = Counter()

    for data, retrieval in zip(dataset, rationale_selection):
        assert data["id"] == retrieval["claim_id"]

        # Count all the gold evidence sentences.
        for doc_key, gold_rationales in data["evidence"].items():
            for entry in gold_rationales:
                counts["relevant"] += len(entry["sentences"])

        claim_id = retrieval["claim_id"]

        for doc_id, pred_sentences in retrieval["evidence"].items():
            true_evidence_sets = data["evidence"].get(doc_id) or []

            for pred_sentence in pred_sentences:
                counts["retrieved"] += 1
                if is_correct(pred_sentence, pred_sentences, true_evidence_sets):
                    counts["correct"] += 1

    evaluation_results = compute_f1_score(counts)
    print(evaluation_results)


def evaluate_label_predictions(corpus_path, dataset_path, label_pred_path):
    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(corpus_path)}
    dataset = jsonlines.open(dataset_path)
    label_prediction = jsonlines.open(label_pred_path)

    pred_labels = []
    true_labels = []

    LABELS = {"CONTRADICT": 0, "NOT_ENOUGH_INFO": 1, "SUPPORT": 2}

    for data, prediction in zip(dataset, label_prediction):
        assert data["id"] == prediction["claim_id"]

        if not prediction["labels"]:
            continue

        claim_id = data["id"]
        for doc_id, pred in prediction["labels"].items():
            pred_label = pred["label"]
            true_label = {es["label"] for es in data["evidence"].get(doc_id) or []}
            assert len(true_label) <= 1, "Currently support only one label per doc"
            true_label = next(iter(true_label)) if true_label else "NOT_ENOUGH_INFO"
            pred_labels.append(LABELS[pred_label])
            true_labels.append(LABELS[true_label])

    print(
        f"Accuracy           {round(sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels), 4)}"
    )
    print(
        f'Macro F1:          {f1_score(true_labels, pred_labels, average="macro").round(4)}'
    )

    print(
        f"F1:                {f1_score(true_labels, pred_labels, average=None).round(4)}"
    )
    print(
        f"Precision:         {precision_score(true_labels, pred_labels, average=None).round(4)}"
    )
    print(
        f"Recall:            {recall_score(true_labels, pred_labels, average=None).round(4)}"
    )
    print()
