import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger

from argparse import ArgumentParser, Namespace
from datetime import datetime

from data.data_module import MMDataModule
from models.main_model import MMClassifier

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def cli_main(hparams: Namespace) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    seed_everything(hparams.seed)
    
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------
    data = MMDataModule(hparams)
    
    hparams.n_classes = data.n_classes
    hparams.labels = data.labels
    hparams.label_freqs = data.label_freqs
    hparams.train_data_len = data.train_data_len
    hparams.vocab = data.vocab
    
    '''
    data.setup("fit")
    data.setup("test")
    
    train_loader = data.train_dataloader()
    #val_loader = data.val_dataloader()
    #test_loader = data.test_dataloader()
    
    text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor = next(iter(train_loader))
    
    print("text_tensor: ", text_tensor.shape)
    print("segment_tensor: ", segment_tensor.shape)
    print("mask_tensor: ", mask_tensor.shape)
    #print("img_tensor: ", img_tensor.shape)
    print("tgt_tensor: ", tgt_tensor.shape)
    '''
        
    model = MMClassifier(hparams)
    
    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        #monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )
    
    # ------------------------
    # 3 INIT LOGGERS
    # ------------------------
    # Tensorboard Callback
    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join(
        "experiments/", tb_logger.version, "checkpoints",
    )

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
        save_weights_only=True
    )

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        gradient_clip_val=1.0,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------
    trainer.fit(model, data)

if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = ArgumentParser(
        description="Multimodal Classifier",
        add_help=True,
    )
    parser.add_argument("--batch_sz", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")
    parser.add_argument("--n_classes", type=int, default=23)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--monitor", default="val_checkpoint_on", type=str)
    parser.add_argument("--metric_mode", default="max", type=str, choices=["auto", "min", "max"])
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="bert", choices=["bow", "img", "bert", "concatbow", "concatbow16", "concatbert", "mmbt", "gmu", "mmtr", "mmbtp", "mmdbt", "vilbert", "mmbt3", "mmvilbt"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "mpaa"])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )

    # each LightningModule defines arguments relevant to it
    parser = MMClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(hparams)