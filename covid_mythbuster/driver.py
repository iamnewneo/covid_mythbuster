import warnings
from transformers import logging
from covid_mythbuster.data_loader import ComythRationaleDataloader
from covid_mythbuster.model import ComythLabelModel, ComythRationaleModel
from covid_mythbuster.trainer import model_trainer
from covid_mythbuster.config import CONFIG

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def main():
    corpus = "./data/corpus.jsonl"
    train_data_path = "./data/claims_train.jsonl"
    val_data_path = "./data/claims_dev.jsonl"
    model_name = "roberta-large"
    # dataloader = ComythLabelDataloader(
    #     model_name=model_name,
    #     corpus_path=corpus,
    #     train_data_path=train_data_path,
    #     val_data_path=val_data_path,
    # )
    dataloader = ComythRationaleDataloader(
        model_name=model_name,
        corpus_path=corpus,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
    )

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()

    print(f"Train Dataloader length: {len(train_dataloader)}")
    print(f"Val Dataloader length: {len(val_dataloader)}")

    # model = ComythLabelModel(
    #     model_name=model_name,
    #     lr_base=CONFIG["LABEL_PRED"]["LR_BASE"],
    #     lr_linear=CONFIG["LABEL_PRED"]["LR_LINEAR"],
    # )
    model = ComythRationaleModel(
        model_name=model_name,
        lr_base=CONFIG["LABEL_PRED"]["LR_BASE"],
        lr_linear=CONFIG["LABEL_PRED"]["LR_LINEAR"],
    )

    trainer = model_trainer(
        model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader
    )


if __name__ == "__main__":
    main()
