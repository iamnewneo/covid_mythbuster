import jsonlines
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from covid_mythbuster import config
from covid_mythbuster.data_loader import (
    ComythLabelPredictionDataset,
    ComythRationaleSelectionDataset,
)


class ComythRationaleDataloader:
    def __init__(self, model_name, corpus_path, train_data_path, val_data_path) -> None:
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

    def get_train_dataloader(
        self, batch_size=config.BATCH_SIZE, num_workers=config.N_WORKER, *args, **kwargs
    ):
        corpus = {doc["doc_id"]: doc for doc in jsonlines.open(self.corpus_path)}
        claims = jsonlines.open(self.train_data_path)
        claims = list(claims)
        if config.ENV == "dev":
            claims = claims[:3]
        train_dataset = ComythRationaleSelectionDataset(
            model_name=self.model_name, corpus=corpus, claims=claims
        )
        return DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            *args,
            **kwargs
        )

    def get_val_dataloader(
        self, batch_size=config.BATCH_SIZE, num_workers=config.N_WORKER, *args, **kwargs
    ):
        corpus = {doc["doc_id"]: doc for doc in jsonlines.open(self.corpus_path)}
        claims = jsonlines.open(self.val_data_path)
        claims = list(claims)
        if config.ENV == "dev":
            claims = claims[:3]
        val_dataset = ComythRationaleSelectionDataset(
            model_name=self.model_name, corpus=corpus, claims=claims
        )
        return DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            *args,
            **kwargs
        )


class ComythLabelDataloader:
    def __init__(self, model_name, corpus_path, train_data_path, val_data_path) -> None:
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

    def get_train_dataloader(
        self, batch_size=config.BATCH_SIZE, num_workers=config.N_WORKER, *args, **kwargs
    ):
        corpus = {doc["doc_id"]: doc for doc in jsonlines.open(self.corpus_path)}
        claims = jsonlines.open(self.train_data_path)
        claims = list(claims)
        if config.ENV == "dev":
            claims = claims[:3]
        train_dataset = ComythLabelPredictionDataset(
            model_name=self.model_name, corpus=corpus, claims=claims
        )
        return DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            *args,
            **kwargs
        )

    def get_val_dataloader(
        self, batch_size=config.BATCH_SIZE, num_workers=config.N_WORKER, *args, **kwargs
    ):
        corpus = {doc["doc_id"]: doc for doc in jsonlines.open(self.corpus_path)}
        claims = jsonlines.open(self.val_data_path)
        claims = list(claims)
        if config.ENV == "dev":
            claims = claims[:3]
        val_dataset = ComythLabelPredictionDataset(
            model_name=self.model_name, corpus=corpus, claims=claims
        )
        return DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            *args,
            **kwargs
        )
