import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_pretrained_bert import BertAdam
from sklearn.metrics import f1_score, accuracy_score
from pytorch_lightning.metrics.functional import f1_score
from pytorch_lightning.metrics.functional import accuracy
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from utils.get_model import get_model


class MMClassifier(pl.LightningModule):

    def __init__(self, hparams: Namespace) -> None:
        super(MMClassifier, self).__init__()
        self.hparams = hparams
        
        self.model = get_model(self.hparams)
        
        if hparams.task_type == "multilabel":
            if hparams.weight_classes:
                freqs = [hparams.label_freqs[l] for l in hparams.labels]
                label_weights = (torch.FloatTensor(freqs) / hparams.train_data_len) ** -1
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        txt, segment, mask, img, y = batch
        
        if self.hparams.model == "bert":
            y_hat = self(txt, mask, segment)
        elif self.hparams.model == "mmbt":
            y_hat = self(txt, mask, segment, img)
        else:
            raise ValueError(f'Specified model ({hparams.model}) not implemented')
        
        loss = self.criterion(y_hat, y)
        
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        
        return result

    def validation_step(self, batch, batch_idx):
        txt, segment, mask, img, y = batch

        if self.hparams.model == "bert":
            y_hat = self(txt, mask, segment)
        elif self.hparams.model == "mmbt":
            y_hat = self(txt, mask, segment, img)
        else:
            raise ValueError(f'Specified model ({hparams.model}) not implemented')
        
        loss = self.criterion(y_hat, y)
        
        if self.hparams.task_type == "multilabel":
            preds = (torch.sigmoid(y_hat) > 0.5).float().detach()
        else:
            preds = torch.nn.functional.softmax(y_hat, dim=1).argmax(dim=1).cpu().detach().numpy()
                    
        if self.hparams.task_type == "multilabel":
            macro_f1 = f1_score(preds, y, class_reduction='macro')
            micro_f1 = f1_score(preds, y, class_reduction='micro')
            
            result = pl.EvalResult(checkpoint_on=micro_f1)
            result.log('val_micro_f1', micro_f1, prog_bar=True, on_epoch=True)
            result.log('val_macro_f1', macro_f1, prog_bar=True, on_epoch=True)
        else:
            acc = accuracy(preds, y)
            result.log('val_acc', acc, prog_bar=True, on_epoch=True)
        result.log('val_loss', loss, prog_bar=True, on_epoch=True)

        return result

    def test_step(self, batch, batch_idx):
        txt, segment, mask, img, y = batch

        if self.hparams.model == "bert":
            y_hat = self(txt, mask, segment)
        elif self.hparams.model == "mmbt":
            y_hat = self(txt, mask, segment, img)
        else:
            raise ValueError(f'Specified model ({hparams.model}) not implemented')
        
        loss = self.criterion(y_hat, y)
        
        if self.hparams.task_type == "multilabel":
            #preds = torch.sigmoid(y_hat).cpu().detach().numpy() > 0.5
            preds = (torch.sigmoid(y_hat) > 0.5).float().detach()
        else:
            preds = torch.nn.functional.softmax(y_hat, dim=1).argmax(dim=1).cpu().detach().numpy()
            
        if self.hparams.task_type == "multilabel":
            macro_f1 = f1_score(preds, y, class_reduction='macro')
            micro_f1 = f1_score(preds, y, class_reduction='micro')
            
            result = pl.EvalResult(checkpoint_on=micro_f1)
            result.log('test_micro_f1', micro_f1, prog_bar=True, on_epoch=True)
            result.log('test_macro_f1', macro_f1, prog_bar=True, on_epoch=True)
        else:
            acc = accuracy(preds, y)
            result.log('test_acc', acc, prog_bar=True, on_epoch=True)
        result.log('test_loss', loss, prog_bar=True, on_epoch=True)

        return result
    
    def exclude_from_wt_decay(self, named_params, skip_list):
        
        optimizer_grouped_parameters = [
            {"params": [p for n, p in named_params if not any(nd in n for nd in skip_list)], "weight_decay": 0.01},
            {"params": [p for n, p in named_params if any(nd in n for nd in skip_list)], "weight_decay": 0.0,},
        ]
        
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        if self.hparams.model in ["bert", "concatbert", "mmbt"]:
            total_steps = (
                self.hparams.train_data_len
                / self.hparams.batch_sz
                / self.hparams.gradient_accumulation_steps
                * self.hparams.max_epochs
            )

            param_optimizer = self.exclude_from_wt_decay(list(self.named_parameters()),
                                                    ["bias", "LayerNorm.bias", "LayerNorm.weight"])
            
            optimizer = BertAdam(
                param_optimizer,
                lr=self.hparams.lr,
                warmup=self.hparams.warmup,
                t_total=total_steps,
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
            
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=self.hparams.lr_patience, verbose=True, factor=self.hparams.lr_factor,
        )
        
        scheduler = {
            'scheduler': scheduler,
            'monitor': 'val_checkpoint_on',
            'interval': 'epoch',
            'frequency': self.hparams.lr_patience
        }
        
        return [optimizer], [scheduler]
    
    @classmethod
    def add_model_specific_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: argparse.ArgumentParser
        Returns:
            - updated parser
        """
        parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
        parser.add_argument("--hidden_sz", type=int, default=768)
        parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "att", "cls_att", "vert_att"], help='Type of pooling technique for BERT models')
        parser.add_argument("--img_hidden_sz", type=int, default=2048)
        parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
        parser.add_argument("--dropout", type=float, default=0.1)
        return parser