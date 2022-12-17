from utils.preprocessor_class import PreprocessorClass
from models.multi_class_model import MultiClassModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    dm = PreprocessorClass(
        preprocessed_dir = "BERT-Classification/data/preprocessed",
        batch_size = 10,
        max_length = 100
    )

    model = MultiClassModel(
        n_out = 5,
        dropout = 0.3,
        lr = 1e-5     # 0.000001
    )

    logger = TensorBoardLogger("logs", name = "bert-multi-class")

    trainer = pl.Trainer(
        # gpus = 1,
        accelerator = "gpu",
        max_epochs = 10,
        default_root_dir = "BERT-Classification/checkpoint/class"
    )

    trainer.fit(model, datamodule = dm)
    pred, true = trainer.predict(model = model, datamodule = dm)