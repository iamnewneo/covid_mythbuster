import random
import jsonlines
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ComythRationaleSelectionDataset(Dataset):
    def __init__(self, model_name, corpus, claims, max_length=512):
        """
        Rationale Selection
        For every claim:
        1. Retrieve document from corpus
        2. Create list of sentences from evidence dictionary
        3. Iterate over every senetce in doc abstract
        4. Mark evidence true or false based on list of setences in evidence
        
        Claim:
        Abstract Stence 1: True
        Abstract Stence 2: False
        Abstract Stence 3: True
        """
        self.samples = []
        self.corpus = corpus
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        for claim in claims:
            for doc_id, evidence in claim["evidence"].items():
                doc = corpus[int(doc_id)]
                evidence_sentence_idx = {s for es in evidence for s in es["sentences"]}
                for i, sentence in enumerate(doc["abstract"]):
                    self.samples.append(
                        self.get_input_dict(
                            claim=claim["claim"],
                            sentence=sentence,
                            evidence=int(i in evidence_sentence_idx),
                        )
                    )

    def get_input_dict(self, claim, sentence, evidence):
        sentence = " ".join(sentence)
        encoding = self.tokenizer.encode_plus(
            sentence,
            claim,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "claim": claim,
            "sentence": sentence,
            "y": evidence,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ComythLabelPredictionDataset(Dataset):
    """
    Label Prediction
    For every claim:
        ## Adding individual evidence here
        For every evidence:
            For every sentence_id in evidence:
                Get abdsttract sentece from doc
                and Add it to dataset with label
                provided in evicdence
        ## Concat All evidence sentences and 
        add it to dataset
        ## Add Not Enough Info
    """

    def __init__(self, model_name, corpus, claims, max_length=512):
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        label_encodings = {"CONTRADICT": 0, "NOT_ENOUGH_INFO": 1, "SUPPORT": 2}

        for claim in claims:
            if claim["evidence"]:
                for doc_id, evidence_sets in claim["evidence"].items():
                    doc = corpus[int(doc_id)]

                    # Add individual evidence set as samples:
                    for evidence_set in evidence_sets:
                        rationale = [
                            doc["abstract"][i].strip()
                            for i in evidence_set["sentences"]
                        ]
                        self.samples.append(
                            self.get_input_dict(
                                claim["claim"],
                                rationale,
                                label_encodings[evidence_set["label"]],
                            )
                        )

                    # Add all evidence sets as positive samples
                    rationale_idx = {s for es in evidence_sets for s in es["sentences"]}
                    rationale_sentences = [
                        doc["abstract"][i].strip() for i in sorted(list(rationale_idx))
                    ]
                    # directly use the first evidence set label
                    # because currently all evidence sets have
                    # the same label
                    self.samples.append(
                        self.get_input_dict(
                            claim["claim"],
                            rationale_sentences,
                            label_encodings[evidence_sets[0]["label"]],
                        )
                    )

                    # Add negative samples
                    non_rationale_idx = set(range(len(doc["abstract"]))) - rationale_idx
                    non_rationale_idx = random.sample(
                        non_rationale_idx,
                        k=min(random.randint(1, 2), len(non_rationale_idx)),
                    )
                    non_rationale_sentences = [
                        doc["abstract"][i].strip()
                        for i in sorted(list(non_rationale_idx))
                    ]
                    self.samples.append(
                        self.get_input_dict(
                            claim["claim"],
                            non_rationale_sentences,
                            label_encodings["NOT_ENOUGH_INFO"],
                        )
                    )
            else:
                # Add negative samples
                for doc_id in claim["cited_doc_ids"]:
                    doc = corpus[int(doc_id)]
                    non_rationale_idx = random.sample(
                        range(len(doc["abstract"])), k=random.randint(1, 2)
                    )
                    non_rationale_sentences = [
                        doc["abstract"][i].strip() for i in non_rationale_idx
                    ]
                    self.samples.append(
                        self.get_input_dict(
                            claim["claim"],
                            non_rationale_sentences,
                            label_encodings["NOT_ENOUGH_INFO"],
                        )
                    )

    def get_input_dict(self, claim, rationale, label):
        rationale = " ".join(rationale)
        encoding = self.tokenizer.encode_plus(
            rationale,
            claim,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "claim": claim,
            "rationale": rationale,
            "y": label,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
