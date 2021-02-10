#!/usr/bin/env python
"""Train a model using the same routine as used in the GNN Benchmarks dataset."""
import argparse
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.utilities import rank_zero_info
import topognn.data_utils as topo_data
from topognn.train_model import MODEL_MAP
from pytorch_lightning.utilities.seed import seed_everything
from topognn.cli_utils import str2bool


class StopOnMinLR(Callback):
    """Callback to stop training as soon as the min_lr is reached.

    This is to mimic the training routine from the publication
    `Benchmarking Graph Neural Networks, V. P. Dwivedi, K. Joshi et al.`
    """

    def __init__(self, min_lr):
        super().__init__()
        self.min_lr = min_lr

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        """Check if lr is lower than min_lr.

        This is the closest to after the update of the lr where we can
        intervene via callback. The lr logger also uses this hook to log
        learning rates.
        """
        for scheduler in trainer.lr_schedulers:
            opt = scheduler['scheduler'].optimizer
            param_groups = opt.param_groups
            for pg in param_groups:
                lr = pg.get('lr')
                if lr < self.min_lr:
                    trainer.should_stop = True
                    rank_zero_info(
                        'lr={} is lower than min_lr={}. '
                        'Stopping training.'.format(lr, self.min_lr)
                    )


def get_logger(name, args):
    if name == 'wandb':
        return WandbLogger(
            name=f"{args.model}_{args.dataset}",
            project="topo_gnn",
            entity="topo_gnn",
            log_model=True,
            tags=[args.model, args.dataset]
        )
    if name == 'tensorboard':
        return TensorBoardLogger(
            'logs',
            name=f"{args.model}_{args.dataset}"
        )
    else:
        raise NotImplementedError()


def main(model_cls, dataset_cls, args):
    args.training_seed = seed_everything(args.training_seed)
    # Instantiate objects according to parameters
    dataset = dataset_cls(**vars(args))
    dataset.prepare_data()

    logger_name = args.logger
    del args.logger  # This does not need to be tracked as a hyperparameter
    model = model_cls(
        **vars(args),
        num_node_features=dataset.node_attributes,
        num_classes=dataset.num_classes,
        task=dataset.task
    )
    print('Running with hyperparameters:')
    print(model.hparams)

    # Loggers and callbacks
    logger = get_logger(logger_name, args)
    log_dir = getattr(logger, 'log_dir', False) or logger.experiment.dir
    stop_on_min_lr_cb = StopOnMinLR(args.min_lr)
    lr_monitor = LearningRateMonitor('epoch')
    checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )

    GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger=logger,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[stop_on_min_lr_cb, checkpoint_cb, lr_monitor]
    )
    trainer.fit(model, datamodule=dataset)
    test_results = trainer.test(test_dataloaders=dataset.test_dataloader())[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys())
    parser.add_argument('--dataset', type=str,
                        choices=topo_data.dataset_map_dict().keys())
    parser.add_argument('--training_seed', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument("--paired", type=str2bool, default=False)
    parser.add_argument("--merged", type=str2bool, default=False)
    parser.add_argument(
        '--logger', choices=['wandb', 'tensorboard'], default='tensorboard')

    partial_args, _ = parser.parse_known_args()

    if partial_args.model is None or partial_args.dataset is None:
        parser.print_usage()
        sys.exit(1)
    model_cls = MODEL_MAP[partial_args.model]
    dataset_cls = topo_data.get_dataset_class(**vars(partial_args))

    parser = model_cls.add_model_specific_args(parser)
    parser = dataset_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, dataset_cls, args)
