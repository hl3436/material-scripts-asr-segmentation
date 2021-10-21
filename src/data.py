from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import json
import torch


class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.fields = list(data.keys())

    def __getitem__(self, idx):
        item = {}
        for field in self.fields:
            if self.data[field] is not None:
                item[field] = self.data[field][idx]
        return item

    def __len__(self):
        return len(self.data[self.fields[0]])


class DataModule(pl.LightningDataModule):
    def __init__(self, args, vocab, pos_vocab=None, dep_vocab=None):
        super().__init__()
        self.args = args

        self.fields = ["tokens", "labels", "prev_labels", "pos"]
        self.padding = {
            "tokens": vocab.pad(),
            "prev_labels": 2,
            "labels": 2,
            "pos": pos_vocab.pad() if pos_vocab is not None else None,
            "dep": dep_vocab.pad() if dep_vocab is not None else None,
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.load_data(self.args.train_path)
            self.valid = self.load_data(self.args.valid_path)
            print(
                "Train loaded: {}".format(
                    ",".join(
                        [
                            "{}: {}".format(field, len(self.train.data[field]))
                            for field in self.train.fields
                        ]
                    )
                )
            )
            print(
                "Valid loaded: {}".format(
                    ",".join(
                        [
                            "{}: {}".format(field, len(self.valid.data[field]))
                            for field in self.valid.fields
                        ]
                    )
                )
            )

        if stage == "test" or stage is None:
            self.test = self.load_data(self.args.test_path)
            print(
                "Test loaded: {}".format(
                    ",".join(
                        [
                            "{}: {}".format(field, len(self.test.data[field]))
                            for field in self.test.fields
                        ]
                    )
                )
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def load_data(self, path):
        data = {}
        # ignore fields that are not present for Dataset
        for field in self.fields:
            # if os.path.isfile("{}.{}.json".format(path, field)):
            #    with open("{}.{}.json".format(path, field), "r") as f:
            if os.path.isfile("{}.{}.npy".format(path, field)):
                data[field] = np.load(
                    "{}.{}.npy".format(path, field), allow_pickle=True
                )
        return MyDataSet(data)

    def collate_fn(self, batch):
        _batch = {}
        for field in self.fields:
            if field in batch[0]:
                data = [b[field] for b in batch]
                _batch[field] = pad_sequence(
                    data, batch_first=True, padding_value=self.padding[field]
                )
                # length
                _batch[field + "_length"] = [len(d) for d in data]
            else:
                _batch[field] = None
                _batch[field + "_length"] = None
        return _batch
