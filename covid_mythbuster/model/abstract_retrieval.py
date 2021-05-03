import jsonlines
import numpy as np
from statistics import mean, median
from sklearn.feature_extraction.text import TfidfVectorizer


class AbstractRetrievalModel:
    def __init__(
        self, corpus_path, dataset_path, top_n, min_gram=1, max_gram=2
    ) -> None:
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.top_n = top_n
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(min_gram, max_gram)
        )
        self.data_vectors = None

    def get_prepared_dataset(self):
        corpus = [
            doc["title"] + " ".join(doc["abstract"])
            for doc in jsonlines.open(self.corpus_path)
        ]
        return corpus

    def fit(self):
        corpus = self.get_prepared_dataset()
        self.data_vectors = self.vectorizer.fit_transform(corpus)

    def get_evaluation_metrics(self, doc_ranks):
        median_val = median(doc_ranks)
        mean_val = mean(doc_ranks)
        min_val = min(doc_ranks)
        max_val = max(doc_ranks)
        print(f"Mid reciprocal rank: {median_val}")
        print(f"Avg reciprocal rank: {mean_val}")
        print(f"Min reciprocal rank: {min_val}")
        print(f"Max reciprocal rank: {max_val}")

        result_obj = {
            "median": median_val,
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
        }
        return result_obj

    def evaluate(self):
        corpus = list(jsonlines.open(self.corpus_path))
        dataset = list(jsonlines.open(self.dataset_path))
        doc_ranks = []
        for claim in dataset:
            claim_vector = self.vectorizer.transform([claim["claim"]]).todense()
            doc_scores = np.asarray(self.data_vectors @ claim_vector.T)
            doc_scores = doc_scores.flatten()
            doc_indices_rank = doc_scores.argsort()[::-1].tolist()
            doc_id_rank = [corpus[idx]["doc_id"] for idx in doc_indices_rank]

            for evidence_id in claim["evidence"].keys():
                rank = doc_id_rank.index(int(evidence_id))
                doc_ranks.append(rank)

        evaluation_results = self.get_evaluation_metrics(doc_ranks)

    def predict(self, claim_text):
        corpus = list(jsonlines.open(self.corpus_path))
        claim_vector = self.vectorizer.transform(claim_text).todense()
        doc_scores = np.asarray(self.data_vectors @ claim_vector.T).squeeze()
        doc_indices_rank = doc_scores.argsort()[::-1].tolist()
        doc_id_rank = [corpus[idx]["doc_id"] for idx in doc_indices_rank]
        abstracts = doc_id_rank[: self.top_n]
        return abstracts
