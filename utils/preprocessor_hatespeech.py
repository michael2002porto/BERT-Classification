import re
import torch
import pytorch_lightning as pl
import pandas as pd

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class PreprocessorHatespeech(pl.LightningDataModule):
    def __init__(self, max_length = 100, batch_size = 30) -> None:
        super(PreprocessorHatespeech, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.max_length = max_length
        self.batch_size = batch_size

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()
        # Menghilangkan imbuhan
        return self.stemmer.stem(string)

    def load_data(self,):
        data = pd.read_csv("data/re_dataset.csv", encoding="latin-1")
        data = data.dropna(how = "any")

        # Ambil nama kolom
        self.hspc_label = list(data.keys())[1:]

        # Buat label ke id untuk variable y
        self.label2id = {}
        for i_hspc, k_hspc in enumerate(self.hspc_label):
            self.label2id[k_hspc] = i_hspc
        # print(self.label2id)

        # Mengambil id baris yang tidak memiliki label
        condition_empty_label = data[
            (
                (data['HS'] == 0) &
                (data['Abusive'] == 0) &
                (data['HS_Individual'] == 0) &
                (data['HS_Group'] == 0) &
                (data['HS_Religion'] == 0) &
                (data['HS_Race'] == 0) &
                (data['HS_Physical'] == 0) &
                (data['HS_Gender'] == 0) &
                (data['HS_Other'] == 0) &
                (data['HS_Weak'] == 0) &
                (data['HS_Moderate'] == 0) &
                (data['HS_Strong'] == 0)
            )
        ].index

        data = data.drop(condition_empty_label)

        tweet = data["Tweet"].apply(lambda x: self.clean_str(x))
        tweet = tweet.values.tolist()
        # print(tweet)

        labels = data.drop(["Tweet"], axis = 1)
        labels = labels.values.tolist()
        # print(labels)

        print(self.hspc_label)

if __name__ == '__main__':
    pre = PreprocessorHatespeech()
    pre.load_data()