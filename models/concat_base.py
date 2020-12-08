import argparse
import configparser
import logging
import warnings
import tempfile
import fasttext
import torch
import json
import os
import random
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torchvision
import pandas_path
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset import HatefulMemesDataset

logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
config = configparser.ConfigParser()
config.read('../config/config.ini')


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
            self,
            num_classes,
            loss_fn,
            language_module,
            vision_module,
            language_feature_dim,
            vision_feature_dim,
            fusion_output_size,
            dropout_p,
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size,
            out_features=num_classes
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, text, image, label=None):
        text_features = torch.nn.functional.relu(self.language_module(text))
        image_features = torch.nn.functional.relu(self.vision_module(image))
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(
            torch.nn.functional.relu(
                self.fusion(combined)
            )
        )
        logits = self.fc(fused)
        pred = torch.nn.functional.softmax(logits)
        loss = (
            self.loss_fn(pred, label)
            if label is not None else label
        )

        return pred, loss


class HatefulMemesModel(pl.LightningModule):
    def __init__(self, hparams):
        for data_key in ["train_path", "dev_path", "img_dir", ]:
            # ok, there's one for-loop but it doesn't count
            if data_key not in hparams.keys():
                raise KeyError(
                    f"{data_key} is a required hparam in this model"
                )

        super(HatefulMemesModel, self).__init__()
        self.hparams = hparams

        # assign some hparams that get used in multiple places
        self.embedding_dim = self.hparams.get("embedding_dim", 300)
        self.language_feature_dim = self.hparams.get("language_feature_dim", 300)
        self.vision_feature_dim = self.hparams.get("vision_feature_dim", self.language_feature_dim)
        self.output_path = Path(self.hparams.get("output_path", "model-outputs"))
        self.output_path.mkdir(exist_ok=True)

        # instantiate transforms, datasets
        self.text_transform = self._build_text_transform()
        self.image_transform = self._build_image_transform()
        self.train_dataset = self._build_dataset("train_path")
        self.dev_dataset = self._build_dataset("dev_path")

        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_nb):
        preds, loss = self.forward(
            text=batch["text"],
            image=batch["image"],
            label=batch["label"]
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        preds, loss = self.eval().forward(
            text=batch["text"],
            image=batch["image"],
            label=batch["label"]
        )

        return {"batch_val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            tuple(
                output["batch_val_loss"]
                for output in outputs
            )
        ).mean()

        return {
            "val_loss": avg_loss,
            "progress_bar": {"avg_val_loss": avg_loss}
        }

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.get("lr", 0.001)
            )
        ]
        schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizers[0]
            )
        ]
        return optimizers, schedulers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16)
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16)
        )

    def fit(self):
        self._set_seed(self.hparams.get("random_state", 42))
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_text_transform(self):
        with tempfile.NamedTemporaryFile() as ft_training_data:
            ft_path = Path(ft_training_data.name)
            with ft_path.open("w") as ft:
                training_data = [
                    json.loads(line)["text"] + "/n"
                    for line in open(
                        self.hparams.get("train_path")
                    ).read().splitlines()
                ]
                for line in training_data:
                    ft.write(line + "\n")
                language_transform = fasttext.train_unsupervised(
                    str(ft_path),
                    model=self.hparams.get("fasttext_model", "cbow"),
                    dim=self.embedding_dim
                )
        return language_transform

    def _build_image_transform(self):
        image_dim = self.hparams.get("image_dim", 224)
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(image_dim, image_dim)),
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/docs/stable/torchvision/models.html
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return image_transform

    def _build_dataset(self, dataset_key):
        return HatefulMemesDataset(
            data_path=self.hparams.get(dataset_key, dataset_key),
            img_dir=self.hparams.get("img_dir"),
            image_transform=self.image_transform,
            text_transform=self.text_transform,
            # limit training samples only
            dev_limit=(
                self.hparams.get("dev_limit", None)
                if "train" in str(dataset_key) else None
            ),
            balance=True if "train" in str(dataset_key) else False,
        )

    def _build_model(self):
        # we're going to pass the outputs of our text
        # transform through an additional trainable layer
        # rather than fine-tuning the transform
        language_module = torch.nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.language_feature_dim
        )

        # easiest way to get features rather than
        # classification is to overwrite last layer
        # with an identity transformation, we'll reduce
        # dimension using a Linear layer, resnet is 2048 out
        vision_module = torchvision.models.resnet152(pretrained=True)
        vision_module.fc = torch.nn.Linear(
            in_features=2048,
            out_features=self.vision_feature_dim
        )

        return LanguageAndVisionConcat(
            num_classes=self.hparams.get("num_classes", 2),
            loss_fn=torch.nn.CrossEntropyLoss(),
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get(
                "fusion_output_size", 512
            ),
            dropout_p=self.hparams.get("dropout_p", 0.1),
        )

    def _get_trainer_params(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=self.output_path,
            monitor=self.hparams.get("checkpoint_monitor", "avg_val_loss"),
            mode=self.hparams.get("checkpoint_monitor_mode", "min"),
            verbose=self.hparams.get("verbose", True)
        )
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=self.hparams.get("early_stop_monitor", "avg_val_loss"),
            min_delta=self.hparams.get("early_stop_min_delta", 0.001),
            patience=self.hparams.get("early_stop_patience", 3),
            verbose=self.hparams.get("verbose", True),
        )
        trainer_params = {
            "checkpoint_callback": checkpoint_callback,
            "early_stop_callback": early_stop_callback,
           #"default_save_path": self.output_path,
            "accumulate_grad_batches": self.hparams.get("accumulate_grad_batches", 1),
            "gpus": self.hparams.get("n_gpu", 1),
            "max_epochs": self.hparams.get("max_epochs", 100),
            "gradient_clip_val": self.hparams.get("gradient_clip_value", 1),
        }

        return trainer_params

    @torch.no_grad()
    def make_submission_frame(self, test_path):
        test_dataset = self._build_dataset(test_path)
        submission_frame = pd.DataFrame(
            index=test_dataset.samples_frame.id,
            columns=["proba", "label"]
        )
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16))
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().to("cpu")(batch["text"], batch["image"])
            submission_frame.loc[batch["id"], "proba"] = preds[:, 1]
            submission_frame.loc[batch["id"], "label"] = preds.argmax(dim=1)
        submission_frame.proba = submission_frame.proba.astype(float)
        submission_frame.label = submission_frame.label.astype(int)
        return submission_frame


def main(chk_path, name, num_epochs):
    hparams = {
        "train_path": '/storage/scratch2/share/eb_research_pool/mmf_dataset/train.jsonl',
        "dev_path": '/storage/scratch2/share/eb_research_pool/mmf_dataset/dev.jsonl',
        "test_path": '/storage/scratch2/share/eb_research_pool/mmf_dataset/test.jsonl',
        "img_dir": '/storage/scratch2/share/eb_research_pool/mmf_dataset/',
        "embedding_dim": 150,
        "language_feature_dim": 300,
        "vision_feature_dim": 300,
        "fusion_output_size": 256,
        "dev_limit": None,
        "lr": 0.00005,
        "max_epochs": num_epochs,
        "output_path": '',
        "n_gpu": 1,
        "batch_size": 4,
        "accumulate_grad_batches": 4,
        "early_stop_patience": 4
    }

    model = HatefulMemesModel(hparams=hparams)
    model.fit()

    save_chk_path = chk_path + name + ".ckpt"
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint, save_chk_path)

    model.test(hparams['test_path'], name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c",
                        default='',
                        help="Path to the checkpoint")
    parser.add_argument("--name", "-n",
                        help="Name of test predictions file")
    parser.add_argument("--epoch", "-e",
                        default=config.getint('model', 'num_epochs'),
                        help="Specifies the number of epochs to train for")
    parser.add_argument("--cuda", "-g",
                        default=0,
                        help="Specifies the gpu")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    main(args.ckpt, args.name, int(args.epoch))
