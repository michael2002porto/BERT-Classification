from utils.preprocessor_class import PreprocessorClass
from models.multi_class_model import MultiClassModel

from sklearn.metrics import accuracy_score

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    dm = PreprocessorClass(
        preprocessed_dir = "data/preprocessed",
        batch_size = 200,
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
        max_epochs = 1,
        default_root_dir = "../checkpoint/class",
        logger = logger
    )

    trainer.fit(model, datamodule = dm)
    hasil = trainer.predict(model = model, datamodule = dm)
    
    # hasil_prediksi = hasil["predictions"][0].tolist()
    # label_asli = hasil["labels"][1].tolist()

    # print(label_asli)
    # print("="*50)
    # print(hasil_prediksi)

    # akurasi = accuracy_score(label_asli, hasil_prediksi)
    # print(akurasi)