# data_cls.py  (새 파일)
from torch.utils.data import Dataset
import torch

class DataPrecessForClassification(Dataset):
    def __init__(self, tokenizer, df, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts  = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()   # 0 / 1
        self._encode()

    def _encode(self):
        self.inputs, self.masks, self.types = [], [], []
        for t in self.texts:
            tokens = ["[CLS]"] + self.tokenizer.tokenize(t)[: self.max_seq_len - 1]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            pad_len = self.max_seq_len - len(ids)
            self.inputs.append(ids + [0] * pad_len)
            self.masks.append([1] * len(ids) + [0] * pad_len)
            self.types.append([0] * self.max_seq_len)

        self.inputs  = torch.tensor(self.inputs,  dtype=torch.long)
        self.masks   = torch.tensor(self.masks,   dtype=torch.long)
        self.types   = torch.tensor(self.types,   dtype=torch.long)
        self.labels  = torch.tensor(self.labels,  dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.inputs[idx],
                self.masks[idx],
                self.types[idx],
                self.labels[idx])
