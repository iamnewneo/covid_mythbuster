import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from covid_mythbuster import config


def model_trainer(model, train_dataloader, val_dataloader, progress_bar_refresh_rate=0):
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, mode="auto"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="./lightning_checkpoints/", save_last=True
    )
    gpus = None
    accelerator = None
    precision = 32
    if config.DEVICE in ["gpu", "cuda", "cuda:0"]:
        gpus = 1
        n_gpus = config.N_GPU
        precision = config.FP_PRECISION
        if n_gpus > 1:
            gpus = list(range(16))[-n_gpus:]
            accelerator = "dp"

    resume_from_checkpoint = None
    if config.RESUME_TRAINING:
        resume_from_checkpoint = "./lightning_checkpoints/last.ckpt"

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=config.MAX_EPOCHS,
        min_epochs=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        weights_summary=None,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        precision=precision,
        accelerator=accelerator,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    # print("Saving Model")
    # torch.save(
    #     trainer.get_model(),
    #     f"{config.BASE_PATH}/models/{model.__class__.__name__}.pth",
    # )
    return trainer
