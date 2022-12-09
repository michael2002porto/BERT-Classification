import random

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

class MultiClassModel(pl.LightningModule):
    def __init__(self, dropout, n_out, lr) -> None:
        super(MultiClassModel, self).__init__()

        torch.manual_seed(1)
        random.seed(1)

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)

        # jumlah label = 5
        self.classifier = nn.Linear(768, n_out)

        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]

        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        
        output = self.classifier(pooler)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids, attention_mask = x_attention_mask, token_type_ids = x_token_type_ids)

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids, attention_mask = x_attention_mask, token_type_ids = x_token_type_ids)
        # ketiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids, attention_mask = x_attention_mask, token_type_ids = x_token_type_ids)
        # ketiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        return pred, true