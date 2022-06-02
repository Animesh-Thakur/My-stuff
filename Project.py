from pickle import TRUE
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchmetrics import Accuracy
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

data_path = "./_ADS/"


class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, csv_file, image_dir):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.tabular = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tabular = self.tabular.iloc[idx, 0:]

        y = tabular["TARGET"]

        image = Image.open(f"{self.image_dir}/{tabular['SUBJECT']}.jpg")
        image = np.array(image)

        image = transforms.functional.to_tensor(image)

        tabular = tabular[["AGE", "SEX"]]
        tabular = tabular.tolist()
        tabular = torch.FloatTensor(tabular)

        return image, tabular, y


def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(
        ), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block


logger = TensorBoardLogger("logs", name="test")


class LitClassifier(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, num_workers: int = 6, batch_size: int = 8,
    ):
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_set = torch.empty(1)

        self.conv1 = conv_block(1, 8)
        self.conv2 = conv_block(8, 32)
        self.conv3 = conv_block(32, 64)

        self.ln1 = nn.Linear(64*3186, 16)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.5)
        self.ln2 = nn.Linear(16, 2)

        self.ln4 = nn.Linear(2, 10)
        self.ln5 = nn.Linear(10, 10)
        self.ln6 = nn.Linear(10, 2)
        self.ln7 = nn.Linear(4, 1)
        self.op8 = nn.Sigmoid()

    def forward(self, img, tab):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = img.reshape(img.shape[0], -1)
        img = self.ln1(img)
        img = self.relu(img)
        img = self.batchnorm(img)
        img = self.dropout(img)
        img = self.ln2(img)
        img = self.relu(img)

        tab = self.ln4(tab)
        tab = self.relu(tab)
        tab = self.ln5(tab)
        tab = self.relu(tab)
        tab = self.ln6(tab)
        tab = self.relu(tab)

        x = torch.cat((img, tab), dim=1)
        x = self.relu(x)

        x = self.ln7(x)

        return self.op8(x)

    def training_step(self, batch, batch_idx):
        image, tabular, y = batch

        criterion = torch.nn.BCELoss()
        y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.double()

        loss = criterion(y_pred, y)

        metric = Accuracy().to('cuda:0')
        y = torch.round(y)
        y_pred = torch.round(y_pred)
        y = y.int()
        y_pred = y_pred.int()
        accuracy = metric(y_pred, y)

        tensorboard_logs = {"train_loss": loss, "train_accuracy": accuracy}
        return {"loss": loss, "accuracy": accuracy, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image, tabular, y = batch

        criterion = torch.nn.BCELoss()
        y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.double()

        val_loss = criterion(y_pred, y)

        metric = Accuracy().to('cuda:0')
        y = torch.round(y)
        y_pred = torch.round(y_pred)
        y = y.int()
        y_pred = y_pred.int()
        accuracy = metric(y_pred, y)

        return {"val_loss": val_loss, "accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        global EPOCH_COUNT
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        writer.add_scalar("Loss/train", avg_loss, EPOCH_COUNT)
        writer.add_scalar("Accuracy/train", avg_acc, EPOCH_COUNT)
        writer.flush()
        tensorboard_logs = {"val_loss": avg_loss, "accuracy": avg_acc}
        EPOCH_COUNT = EPOCH_COUNT + 1
        return {"val_loss": avg_loss, "accuracy": avg_acc, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        image, tabular, y = batch

        criterion = torch.nn.BCELoss()
        y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.double()

        test_loss = criterion(y_pred, y)

        metric = Accuracy().to('cuda:0')
        y = torch.round(y)
        y_pred = torch.round(y_pred)
        y = y.int()
        y_pred = y_pred.int()
        accuracy = metric(y_pred, y)

        return {"test_loss": test_loss, "accuracy": accuracy}

    def test_epoch_end(self, outputs):
        global EPOCH_COUNT
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        writer.add_scalar("Loss/train", avg_loss, EPOCH_COUNT)
        writer.add_scalar("Accuracy/train", avg_acc, EPOCH_COUNT)
        writer.flush()
        logs = {"test_loss": avg_loss, "accuracy": avg_acc}
        EPOCH_COUNT = EPOCH_COUNT + 1
        return {"test_loss": avg_loss, "accuracy": avg_acc, "log": logs, "progress_bar": logs}

    def setup(self, stage):

        image_data = ImageDataset(
            csv_file=f"{data_path}DATA.csv", image_dir=f"{data_path}images/")

        train_size = int(0.833333333333 * len(image_data))
        val_size = int((len(image_data) - train_size) / 2)
        test_size = int((len(image_data) - train_size) / 2)

        self.eval_set = image_data

        self.train_set, self.val_set, self.test_set = random_split(
            image_data, (train_size, val_size, test_size))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image, tabular, y = batch
        y_pred = torch.flatten(self(image, tabular))
        print(y_pred)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.eval_set, batch_size=1)


if __name__ == "__main__":
    # EPOCH_COUNT = 1
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # logger = TensorBoardLogger("logs", name="test")

    # model = LitClassifier().to(device)
    # trainer = pl.Trainer(gpus=1, logger=logger,
    #                      max_epochs=16, auto_select_gpus=TRUE, log_every_n_steps=40)

    # trainer.fit(model)
    # trainer.test(model)
    # torch.save(model.state_dict(), 'model.pth')

    # writer.close()

    model = LitClassifier()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    dataloader = ImageDataset(
        csv_file=f"{data_path}DATA.csv", image_dir=f"{data_path}images/").__getitem__(5)

    trainer = pl.Trainer()
    prediction = trainer.predict(model, dataloader)
