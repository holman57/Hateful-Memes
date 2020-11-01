import argparse
import csv
import json
import random
import tempfile
import datetime
import configparser
import os
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# References for code usage and source material
# https://docs.google.com/document/d/1O_k69fzN9OaRreJYORCln6jmZrbRtZXq-snG0UIBn4M/

start = datetime.datetime.now()
config = configparser.ConfigParser()
config.read('alfred.ini')

def image_batcher(img_path, high_idx):
    img_list, dirpath = [], img_path.split('/raw_images')[0]
    data = json.load(open(dirpath.replace("[", "").replace("'", "") + '/traj_data.json'))
    for img in data['images']:
        if img['high_idx'] == high_idx:
            img_list.append(img_path.replace("[", "").replace("'", "") + "/" + img['image_name'].replace("png", "jpg"))

    while len(img_list) > 50:
        img_list.pop(random.randrange(1, len(img_list)) - 1)

    return img_list


class VLNDataset(Dataset):
    def __init__(self,
                 csv_file,
                 image_transform,
                 text_transform,
                 root_dir="", ):

        self.instance = pd.read_csv(csv_file, encoding="utf-8-sig")#, sep='|')
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        iid = self.instance['id'].iloc[idx]
        img_list = image_batcher(
            os.path.join(self.root_dir, self.instance['images'].iloc[idx]),
            self.instance['high_descs_id'].iloc[idx]
        )

        images = []
        for i in range(len(img_list)):
            images.append(Image.open(os.path.join(self.instance['images'].iloc[idx], img_list[i])).convert("RGB"))
            images[-1] = self.image_transform(images[-1])

        instruction = torch.Tensor(
            self.text_transform.get_sentence_vector(self.instance['instruction'].iloc[idx])).squeeze()

        label = self.instance['label'].iloc[idx]
        sample = {'id': iid, 'images': images, 'text': instruction, 'label': label}

        return sample


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
            self,
            num_classes,
            loss_fn,
            language_module,
            vision_module,
            lstm,
            language_feature_dim,
            vision_feature_dim,
            fusion_output_size,
            dropout_p,
            n_gpu
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.vision_feature_dim = vision_feature_dim
        self.language_module = language_module
        self.vision_module = vision_module
        self.lstm = lstm
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
        self.n_gpu = n_gpu

    def forward(self, text, images, label=None):
        text_features = torch.nn.functional.relu(self.language_module(text.cuda()))
        image_features = []
        for img in images:
            torch.nn.functional.relu(self.vision_module(img.cuda()))
            if self.n_gpu > 0:
                linear = torch.nn.Linear(1000, self.vision_feature_dim).cuda()
                image_features.append(linear(torch.nn.functional.relu(self.vision_module(img.cuda())).cuda()))
            else:
                linear = torch.nn.Linear(1000, self.vision_feature_dim)
                image_features.append(linear(torch.nn.functional.relu(self.vision_module(img))))
        features = torch.stack(image_features, dim=1)
        if self.n_gpu > 0:
            start_size = 16, features.size()[1], 32
            hidden = (torch.randn(start_size).cuda(), torch.randn(start_size).cuda())
            video, _ = self.lstm(features, hidden)
        else:
            video, _ = self.lstm(features)
        video_features = self.lstm.fc(video)[:, -1, :].view(list(video.size())[0], -1)
        combined = torch.cat([text_features, video_features], dim=1)
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


class VLNmodel(pl.LightningModule):
    def __init__(self, hparams):
        for data_key in ["train_path", "dev_path", "img_dir"]:
            if data_key not in hparams.keys():
                raise KeyError(f"{data_key} is a required hparam in this model")

        super(VLNmodel, self).__init__()
        self.hparams = hparams

        # assign some hparams that get used in multiple places
        self.embedding_dim = self.hparams.get(
            "embedding_dim", 300
        )
        self.language_feature_dim = self.hparams.get(
            "language_feature_dim", 300
        )
        self.vision_feature_dim = self.hparams.get(
            # balance language and vision features by default
            "vision_feature_dim", self.language_feature_dim
        )
        self.output_path = Path(
            self.hparams.get(
                "output_path", "model-outputs"
            )
        )
        self.output_path.mkdir(exist_ok=True)
        self.n_gpu = self.hparams.get(
            "n_gpu", 0
        )

        # instantiate transforms, datasets
        self.text_transform = self._build_text_transform()
        self.image_transform = self._build_image_transform()
        self.train_dataset = self._build_dataset("train_path")
        self.dev_dataset = self._build_dataset("dev_path")

        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

    def forward(self, text, images, label=None):
        return self.model(text, images, label)

    def training_step(self, batch, batch_nb):
        preds, loss = self.forward(
            text=batch["text"],
            images=batch["images"],
            label=batch["label"]
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        preds, loss = self.eval().forward(
            text=batch["text"],
            images=batch["images"],
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

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 2),
            num_workers=self.hparams.get("num_workers", 8)
        )

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 2),
            num_workers=self.hparams.get("num_workers", 8)
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
                with open(self.hparams.get("train_path"), 'r') as read_obj:
                    csv_dict_reader = csv.DictReader(read_obj)#, delimiter='|')
                    training_data = [
                        row['instruction'] + "/n"
                        for row in csv_dict_reader
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
        image_dim = self.hparams.get("image_dim", 112)
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(image_dim, image_dim)
                ),
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/docs/stable/torchvision/models.html#video-classification
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )

        return image_transform

    def _build_dataset(self, dataset_key):
        return VLNDataset(
            csv_file=self.hparams.get(dataset_key, dataset_key),
            image_transform=self.image_transform,
            text_transform=self.text_transform
        )

    def _build_model(self):
        language_module = torch.nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.language_feature_dim
        )
        vision_module = torchvision.models.resnet152(
            pretrained=True
        )
        lstm = torch.nn.LSTM(
            input_size=self.vision_feature_dim,
            hidden_size=32,
            num_layers=16
        )
        lstm.fc = torch.nn.Linear(
            in_features=32,
            out_features=self.vision_feature_dim
        )

        return LanguageAndVisionConcat(
            num_classes=self.hparams.get(
                "num_classes", 2
            ),
            loss_fn=torch.nn.CrossEntropyLoss(),
            language_module=language_module,
            vision_module=vision_module,
            lstm=lstm,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get(
                "fusion_output_size", 256
            ),
            dropout_p=self.hparams.get(
                "dropout_p", 0.1
            ),
            n_gpu=self.n_gpu
        )

    def _get_trainer_params(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=self.output_path,
            monitor=self.hparams.get(
                "checkpoint_monitor", "avg_val_loss"
            ),
            mode=self.hparams.get(
                "checkpoint_monitor_mode", "min"
            ),
            verbose=self.hparams.get(
                "verbose", True
            )
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=self.hparams.get(
                "early_stop_monitor", "avg_val_loss"
            ),
            min_delta=self.hparams.get(
                "early_stop_min_delta", 0.001
            ),
            patience=self.hparams.get(
                "early_stop_patience", 3
            ),
            verbose=self.hparams.get(
                "verbose", True
            )
        )

        trainer_params = {
            "checkpoint_callback": checkpoint_callback,
            "early_stop_callback": early_stop_callback,
            "default_root_dir": self.output_path,
            "accumulate_grad_batches": self.hparams.get(
                "accumulate_grad_batches", 1
            ),
            "gpus": self.hparams.get("n_gpu", 1),
            "max_epochs": self.hparams.get("max_epochs", 100),
            "gradient_clip_val": self.hparams.get(
                "gradient_clip_value", 1
            ),
            "logger": False
        }

        return trainer_params

    @torch.no_grad()
    def test(self, test_path, name):
        test_dataset = self._build_dataset(test_path)
        output_test_frame = pd.DataFrame(
            index=test_dataset.instance.id,
            columns=["proba", "label"]
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16))
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().cuda()(
                batch["text"], batch["images"]
            )
            output_test_frame.loc[batch["id"], "proba"] = preds[:, 1].cpu()
            output_test_frame.loc[batch["id"], "label"] = preds.argmax(dim=1).cpu()
        output_test_frame.proba = output_test_frame.proba.astype(float)
        output_test_frame.label = output_test_frame.label.astype(int)

        output_save_path = name + ".csv"
        output_test_frame.to_csv(output_save_path)
        try:
            os.chmod(output_save_path, 0o777)
        except:
            pass
        print(output_test_frame)


def main(tr_path, val_path, te_path, chk_path, name, num_epochs):

    hparams = {
        "train_path": tr_path,
        "dev_path": val_path,
        "img_dir": config.get('general', 'alfred_full'),
        "embedding_dim": 150,
        "language_feature_dim": 300,
        "vision_feature_dim": 300,
        "fusion_output_size": 256,
        "dev_limit": None,
        "lr": 0.00005,
        "max_epochs": num_epochs,
        "output_path": config.get('general', 'model_output'),
        "n_gpu": 1,
        "batch_size": 4,
        "accumulate_grad_batches": 4,
        "early_stop_patience": 3
    }


    VLN = VLNmodel(hparams=hparams)
    VLN.fit()

    checkpoint = {}
    checkpoint['state_dict'] = VLN.state_dict()
    chk_save_path = name + ".ckpt"
    torch.save(checkpoint, chk_save_path)
    try:
        os.chmod(chk_save_path, 0o777)
    except:
        pass

    VLN.test(te_path, name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tr_path", help="Path to training instances")
    parser.add_argument("val_path", help="Path validation instances")
    parser.add_argument("te_path", help="Path to test instances")
    parser.add_argument("chk_path", help="Path to the checkpoint")
    parser.add_argument("--cuda", "-c",
                        default="0",
                        help="Specifies the cuda gpu node core")
    parser.add_argument("--name", "-n",
                        help="Name of the output preds")
    parser.add_argument("--epoch", "-e",
                        default=config.getint('model', 'num_epochs'),
                        help="Specifies the number of epochs to train for")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

    main(args.tr_path, args.val_path, args.te_path, args.chk_path, args.name, int(args.epoch))