from argparse import ArgumentParser
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np

import json

import pickle

from .modules import Embedding, Linear, LSTM

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClf(pl.LightningModule):
    def __init__(self, hparams, vocab, pretrained_embedding=None, pos_vocab=None):
        super().__init__()
        self.hparams = hparams

        self.vocab = vocab
        self.inp_dim = len(vocab)

        self.pos_vocab = pos_vocab

        self.embed = Embedding(
            len(vocab),
            embedding_dim=self.hparams.embedding_dim,
            padding_idx=vocab.pad(),
        )

        if self.hparams.label_embedding_dim:
            # 0 and 1, 2 for padding
            self.orig_label_embed = Embedding(
                3, embedding_dim=self.hparams.label_embedding_dim, padding_idx=2
            )

        if self.hparams.pos_dim:
            self.pos_embed = Embedding(
                len(pos_vocab),
                embedding_dim=self.hparams.pos_dim,
                padding_idx=pos_vocab.pad(),
            )

        if pretrained_embedding is not None:
            self.embed = self.load_pretrained_embedding(
                self.embed, self.vocab, self.hparams.pretrained_embedding
            )

        self.num_layers = self.hparams.num_layers
        self.hidden_dim = self.hparams.hidden_dim

        self.bidirectional = getattr(self.hparams, "bidirectional", False)
        self.batch_first = getattr(self.hparams, "batch_first", False)

        lstm_inp_dim = self.hparams.embedding_dim + self.hparams.label_embedding_dim

        if self.hparams.pos_dim:
            lstm_inp_dim += self.hparams.pos_dim

        self.lstm = LSTM(
            input_size=lstm_inp_dim,
            hidden_size=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            batch_first=self.batch_first,
            bidirectional=self.bidirectional,
        )

        lin_inp = (
            self.hparams.hidden_dim * 2
            if self.bidirectional
            else self.hparams.hidden_dim
        )

        self.clf = Linear(lin_inp, 2)
        self.dropout = nn.Dropout(self.hparams.dropout)

        # CRF
        if getattr(self.hparams, "crf", False):
            self.clf = Linear(lin_inp, 3)  # for padding which we will ignore
            self.crf = CRF(3, batch_first=True)

        self.underseg_prob = torch.Tensor(
            [[self.hparams.underseg_prob, 1 - self.hparams.underseg_prob]]
        )

        self.overseg_prob = torch.Tensor(
            [[1 - self.hparams.overseg_prob, self.hparams.overseg_prob]]
        )

        print(self.underseg_prob, self.overseg_prob)

        print(self)

    def forward(self, tokens, tokens_len, labels, labels_len, pos=None, pos_len=None):
        hidden = (
            torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                tokens.size(0),
                self.hidden_dim,
                device=self.device,
            ),
            torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                tokens.size(0),
                self.hidden_dim,
                device=self.device,
            ),
        )

        x = self.embed(tokens)

        if self.hparams.label_embedding_dim:
            lab = self.orig_label_embed(labels)
            x = torch.cat([x, lab], -1)

        if self.hparams.pos_dim:
            pos_embed = self.pos_embed(pos)
            x = torch.cat([x, pos_embed], -1)

        x = self.dropout(x)
        # pack sequence
        x = pack_padded_sequence(x, tokens_len, batch_first=True, enforce_sorted=False)
        x, (h_n, c_n) = self.lstm(x, hidden)
        # unpack
        x, x_len = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)

        return self.clf(x)

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-3)
        return optim

    def generate_prev_labels(self, x, x_len, y, y_len):
        underseg = torch.multinomial(
            self.underseg_prob.repeat(y.size(0), 1), y.size(1), replacement=True
        ).to(x.device)
        underseg_label = torch.where(y == 1, underseg, y).to(x.device)
        overseg = torch.multinomial(
            self.overseg_prob.repeat(y.size(0), 1), y.size(1), replacement=True
        ).to(x.device)
        overseg_label = torch.where(y == 0, overseg, y).to(x.device)

        if self.hparams.prev_label_type == "comb":
            initial_label = torch.where(y == 1, underseg, overseg_label)
            prev_labels = [initial_label]
        elif self.hparams.prev_label_type == "underseg":
            prev_labels = [underseg_label]
        elif self.hparams.prev_label_type == "overseg":
            prev_labels = [overseg_label]
        else:
            prev_labels = [overseg_label, underseg_label]

        return prev_labels

    def training_step(self, batch, batch_idx):
        x, x_len = batch["tokens"], batch["tokens_length"]
        y, y_len = batch["labels"], batch["labels_length"]
        pos = batch["pos"] if "pos" in batch else None

        if self.hparams.label_embedding_dim:
            prev_labels = self.generate_prev_labels(x, x_len, y, y_len)
            logits = [
                self(x, x_len, prev_label, None, pos=pos) for prev_label in prev_labels
            ]
        else:
            logits = [self(x, x_len, None, None, pos=pos)]

        # calc loss
        if getattr(self.hparams, "crf", False):
            mask = y != 2
            # mask = None
            losses = [-self.crf(logit, y, mask=mask) for logit in logits]
        else:
            y = y.view(-1)
            losses = [
                F.cross_entropy(
                    logit.view(-1, logit.shape[-1]), y, ignore_index=2, reduction="sum"
                )
                for logit in logits
            ]

        loss = sum(losses)

        result = pl.TrainResult(loss)
        result.log("train_loss", loss)

        return result

    def validation_step(self, batch, batch_idx):
        x, x_len = batch["tokens"], batch["tokens_length"]
        y, y_len = batch["labels"], batch["labels_length"]
        pos = batch["pos"] if "pos" in batch else None

        if self.hparams.label_embedding_dim:
            prev_labels = self.generate_prev_labels(x, x_len, y, y_len)
            logits = [
                self(x, x_len, prev_label, None, pos=pos) for prev_label in prev_labels
            ]
        else:
            logits = [self(x, x_len, None, None, pos=pos)]

        # calc loss
        if getattr(self.hparams, "crf", False):
            mask = y != 2
            # mask = None
            losses = [-self.crf(logit, y, mask=mask) for logit in logits]
        else:
            y = y.view(-1)
            losses = [
                F.cross_entropy(
                    logit.view(-1, logit.shape[-1]), y, ignore_index=2, reduction="sum"
                )
                for logit in logits
            ]
        loss = sum(losses)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.loss = loss
        result.log("val_loss", loss)

        return result

    def test_step(self, batch, batch_idx):
        x, x_len = batch["tokens"], batch["tokens_length"]
        prev_labels = batch["prev_labels"] if "prev_labels" in batch else None
        y = batch["labels"] if "labels" in batch else None
        pos = batch["pos"] if "pos" in batch else None

        if prev_labels is None:
            # insert 1 to non ending parts, learning undersegmenting problem
            underseg = torch.multinomial(
                self.underseg_prob.repeat(y.size(0), 1), y.size(1), replacement=True
            ).to(x.device)
            initial_label = torch.where(y == 1, underseg, y)
            # underseg_label = torch.where(y == 0, underseg, y).to(x.device)

            # remove 1 from ending parts, creating oversegmenting scenerio
            overseg = torch.multinomial(
                self.overseg_prob.repeat(y.size(0), 1), y.size(1), replacement=True
            ).to(x.device)
            initial_label = torch.where(y == 0, overseg, initial_label)
            prev_labels = initial_label

        logits = self(x, x_len, prev_labels, None, pos=pos)

        # calc loss

        result = pl.EvalResult()

        if y is not None:
            if getattr(self.hparams, "crf", False):
                loss = -self.crf(logits, y, reduction="none")

            else:
                _n = logits.shape[1]
                _y = y.view(-1)
                _logits = logits.view(-1, logits.shape[-1])
                loss = F.cross_entropy(
                    _logits, _y, ignore_index=2, reduction="none"
                ).view(-1, _n)
            result.loss = loss.mean(-1)

        if getattr(self.hparams, "crf", False):
            best_tag_sequence = self.crf.decode(logits)
            confidence = self.crf(
                logits,
                torch.tensor(best_tag_sequence).to(logits.device),
                reduction="none",
            )
            logits = best_tag_sequence
        else:
            logits = logits.argmax(-1)

        result.pred = [line[prev_labels[row] != 2] for row, line in enumerate(logits)]

        return result

    def test_end(self, test_step_outputs):
        result = pl.EvalResult()

        # result.log("loss", test_step_outputs.loss.mean())
        # result.log("f1", np.mean(test_step_outputs.f1))

        if getattr(test_step_outputs, "loss", None) is not None:
            with open("loss.data", "wb") as f:
                pickle.dump(test_step_outputs.loss.data.tolist(), f)

        if getattr(self.hparams, "crf", False):
            pred = [item for sublist in test_step_outputs.pred for item in sublist]
        else:
            # pred = test_step_outputs.pred
            pred = [
                item.data.tolist()
                for sublist in test_step_outputs.pred
                for item in sublist
            ]
            # pred = pred.data.tolist()

        with open("/work/pred.data", "wb") as f:
            pickle.dump(pred, f)

        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_dim", type=int, default=300)
        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=2)

        parser.add_argument("--dropout", type=float, default=0.5)

        parser.add_argument("--batch_first", action="store_true")
        parser.add_argument("--bidirectional", action="store_true")
        parser.add_argument("--crf", action="store_true")

        parser.add_argument("--pos_dim", type=int, default=0)
        parser.add_argument("--dep_dim", type=int, default=0)
        parser.add_argument("--label_embedding_dim", type=int, default=16)

        parser.add_argument("--overseg_prob", type=float, default=0.75)
        parser.add_argument("--underseg_prob", type=float, default=0.75)

        return parser

    def load_pretrained_embedding(self, embed, vocab, pretrained_file):
        # with open(pretrained_file) as f:
        #    pretrained = json.load(f)
        # for i, token in enumerate(vocab.itos):
        #    if token in pretrained:
        #        embed.weight.data[i] = torch.Tensor(pretrained[token])
        embed_dict = dict()
        with open(pretrained_file) as f:
            next(f)  # skip header
            for line in f:
                pieces = line.rstrip().split(" ")
                embed_dict[pieces[0] + "</w>"] = torch.Tensor(
                    [float(weight) for weight in pieces[1:]]
                )
        for i, token in enumerate(vocab.itos):
            if token in embed_dict:
                embed.weight.data[i] = embed_dict[token]
        print("Pretrained embedding loaded")
        return embed
