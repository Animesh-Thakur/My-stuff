from operator import mod
from PIL import Image
import torch
from Project import LitClassifier
from Project import ImageDataset
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np

model = LitClassifier(batch_size=2)
model.load_state_dict(torch.load('model.pth'))
model.eval()


trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.predict(model)
