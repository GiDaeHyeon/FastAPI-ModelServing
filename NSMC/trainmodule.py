import torch.nn as nn
import torch.optim as optim

from torchmetrics import Accuracy
from pytorch_lightning import LightningModule

from NSMC.model import BertModel


class Classifier(LightningModule):
    def __init__(self,
                 model_=None,
                 lr=1e-4):
        super().__init__()
        if model_ is not None:
            self.model = model_
        else:
            self.model = BertModel()
        self.loss_fn = nn.NLLLoss()

        self.accuracy = Accuracy()
        self.lr = lr

    def forward(self, input_ids, attention_masks, token_type_ids):
        return self.model(input_ids, attention_masks, token_type_ids)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, token_type_ids, label = batch
        y_hat = self(input_ids, attention_masks, token_type_ids)
        loss = self.loss_fn(y_hat, label)

        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, token_type_ids, label = batch
        y_hat = self(input_ids, attention_masks, token_type_ids)
        loss = self.loss_fn(y_hat, label)

        self.accuracy(preds=y_hat, target=label)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('Accuracy', self.accuracy, on_step=False, on_epoch=True)
        return {'loss': loss}