import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertModel
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

class MultiClassModel(pl.LightningModule):
    def __init__(self, dropout, n_out, lr):
        super(MultiClassModel, self).__init__()

        torch.manual_seed(1)
        random.seed(1)

        self.num_classes = n_out

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_out)

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.f1 = MulticlassF1Score(average="micro", num_classes=self.num_classes)

    def forward(self, input_ids):
        bert_out = self.bert(input_ids=input_ids)
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, y = train_batch
        out = self(input_ids=x_input_ids)
        f1_score = self.f1(out, y)
        loss = self.criterion(out, y)

        self.log("f1_score", f1_score, prog_bar=True)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, valid_batch, batch_idx):
        print(f"Valid Batch: {valid_batch}")
        x_input_ids, y = valid_batch
        
        out = self(input_ids=x_input_ids)
        loss = self.criterion(out, y)
        f1_score = self.f1(out, y)
        
        self.log("val_f1_score", f1_score, prog_bar=True)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, pred_batch, batch_idx):
        x_input_ids, y = pred_batch
        out = self(input_ids=x_input_ids)
        pred = out
        true = y
        return {"predictions": pred, "labels": true}
