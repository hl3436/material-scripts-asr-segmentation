import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from models import LSTMClf, TransformerClf
from data import DataModule
from vocab import Vocab, POSVocab
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def main(args):
    dict_args = vars(args)

    # load vocab
    vocab = Vocab(args.vocab_file)
    pos_vocab = POSVocab()

    # load data
    dm = DataModule(args, vocab, pos_vocab=pos_vocab)

    checkpoint_callback = ModelCheckpoint(filepath=args.model_path)  # Works with PyTorch 1.7.0

    trainer = Trainer.from_argparse_args(
        args, checkpoint_callback=checkpoint_callback
    )

    if args.model == "lstm":
        if getattr(args, "load_from_checkpoint", False):
            model = LSTMClf.load_from_checkpoint(
                args.load_from_checkpoint, vocab=vocab, pos_vocab=pos_vocab
            )
        else:
            model = LSTMClf(
                args,
                vocab=vocab,
                pretrained_embedding=args.pretrained_embedding,
                pos_vocab=pos_vocab,
            )
    elif args.model == "transformer":
        if getattr(args, "load_from_checkpoint", False):
            model = TransformerClf.load_from_checkpoint(
                args.load_from_checkpoint, vocab=vocab, pos_vocab=pos_vocab
            )
        else:
            model = TransformerClf(
                args,
                vocab=vocab,
                pretrained_embedding=args.pretrained_embedding,
                pos_vocab=pos_vocab,
            )

    # Training
    if getattr(args, "train_path", False):
        dm.setup("fit")
        trainer.fit(model, datamodule=dm)

    # Test
    if getattr(args, "test_path", False):
        dm.setup("test")
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # Data
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)

    # Checkpointing
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--load_from_checkpoint", type=str)

    # Vocab
    parser.add_argument("--vocab_file", type=str)
    parser.add_argument("--pretrained_embedding", type=str)

    parser.add_argument(
        "--model", type=str, default="lstm", choices=["lstm", "transformer"]
    )

    # model
    parser.add_argument(
        "--prev_label_type",
        default="comb",
        choices=["comb", "separate", "underseg", "overseg"],
    )

    temp_args, _ = parser.parse_known_args()
    if temp_args.model == "lstm":
        parser = LSTMClf.add_model_specific_args(parser)
    elif temp_args.model == "transformer":
        parser = TransformerClf.add_model_specific_args(parser)

    args = parser.parse_args()
    print(args)

    main(args)
