import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import pytorch_lightning as pl

from torch_geometric.nn import global_mean_pool, global_add_pool

from topognn import Tasks
from topognn.cli_utils import str2bool
from topognn.layers import GCNLayer, GINLayer, CoordfnTopologyLayer, SetfnTopologyLayer, GFLReadout
from topognn.metrics import WeightedAccuracy


import wandb


class GCNModel(pl.LightningModule):
    def __init__(self, hidden_dim, depth, num_node_features, num_classes, task,
                 lr=0.001, dropout_p=0.2, GIN=False, batch_norm=False,
                 residual=False, train_eps=True, gfl_readout = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        if GIN:
            def build_gnn_layer():
                return GINLayer(
                    hidden_dim, hidden_dim, F.relu, dropout_p, batch_norm,
                    train_eps=train_eps)
            graph_pooling_operation = lambda x,y : global_add_pool(x,y.batch)
        else:
            def build_gnn_layer():
                return GCNLayer(
                    hidden_dim, hidden_dim, F.relu, dropout_p, batch_norm)
            graph_pooling_operation = lambda x,y : global_mean_pool(x,y.batch)

        self.layers = nn.ModuleList([
            build_gnn_layer() for _ in range(depth)])

        if task is Tasks.GRAPH_CLASSIFICATION:
            if gfl_readout:
                self.pooling_fun = GFLReadout(hidden_dim)
            else:
                self.pooling_fun = graph_pooling_operation

        elif task is Tasks.NODE_CLASSIFICATION:
            def fake_pool(x, batch):
                return x
            self.pooling_fun = fake_pool
        else:
            raise RuntimeError('Unsupported task.')

        if (kwargs.get("dim1", False) and ("dim1_out_dim" in kwargs.keys()) and (not kwargs.get("fake", False))):
            dim_before_class = hidden_dim + \
                kwargs["dim1_out_dim"]  # SimpleTopoGNN with dim1
        else:
            if gfl_readout:
                dim_before_class = 200
            else:
                dim_before_class = hidden_dim

        self.classif = torch.nn.Sequential(
            nn.Linear(dim_before_class, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
            )


        self.task = task

        if task is Tasks.GRAPH_CLASSIFICATION:
            self.accuracy = pl.metrics.Accuracy()
            self.accuracy_val = pl.metrics.Accuracy()
            self.accuracy_test = pl.metrics.Accuracy()
            self.loss = torch.nn.CrossEntropyLoss()
        elif task is Tasks.NODE_CLASSIFICATION:
            self.accuracy = WeightedAccuracy(num_classes)
            self.accuracy_val = WeightedAccuracy(num_classes)
            self.accuracy_test = WeightedAccuracy(num_classes)

            def weighted_loss(pred, label):
                # calculating label weights for weighted loss computation
                with torch.no_grad():
                    n_classes = pred.shape[1]
                    V = label.size(0)
                    label_count = torch.bincount(label)
                    label_count = label_count[label_count.nonzero(
                        as_tuple=True)].squeeze()
                    cluster_sizes = torch.zeros(
                        n_classes, dtype=torch.long, device=pred.device)
                    cluster_sizes[torch.unique(label)] = label_count
                    weight = (V - cluster_sizes).float() / V
                    weight *= (cluster_sizes > 0).float()
                return F.cross_entropy(pred, label, weight)

            self.loss = weighted_loss

    def configure_optimizers(self):
        """Reduce learning rate if val_loss doesnt improve."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5,
                patience=self.hparams.lr_patience
            ),
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch"
        }

        return [optimizer], [scheduler]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, edge_index=edge_index, data=data)

        x = self.pooling_fun(x, data)
        x = self.classif(x)

        return x

    def training_step(self, batch, batch_idx):
        y = batch.y
        # Flatten to make graph classification the same as node classification
        y = y.view(-1)
        y_hat = self(batch)
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        loss = self.loss(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y
        # Flatten to make graph classification the same as node classification
        y = y.view(-1)
        y_hat = self(batch)
        y_hat = y_hat.view(-1, y_hat.shape[-1])

        loss = self.loss(y_hat, y)

        self.accuracy_val(y_hat, y)

        self.log("val_loss", loss, on_epoch=True)

        self.log("val_acc", self.accuracy_val, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

        self.accuracy_test(y_hat, y)
        self.log("test_acc", self.accuracy_test, on_epoch=True)

        return {"y": y, "y_hat": y_hat}

    def test_epoch_end(self, outputs):
        y = torch.cat([output["y"] for output in outputs])
        y_hat = torch.cat([output["y_hat"] for output in outputs])

        y_hat_max = torch.argmax(y_hat, 1)
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger):
            # Log confusion matrices
            import wandb
            self.logger.experiment.log({"conf_mat": wandb.plot.confusion_matrix(
                preds=y_hat_max.cpu().numpy(), y_true=y.cpu().numpy())})

    @classmethod
    def add_model_specific_args(cls, parent):
        parser = argparse.ArgumentParser(parents=[parent])
        parser.add_argument("--hidden_dim", type=int, default=146)
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--lr_patience", type=int, default=10)
        parser.add_argument("--min_lr", type=float, default=0.00001)
        parser.add_argument("--dropout_p", type=float, default=0.0)
        parser.add_argument('--GIN', type=str2bool, default=False)
        parser.add_argument('--train_eps', type=str2bool, default=True)
        parser.add_argument('--batch_norm', type=str2bool, default=True)
        parser.add_argument('--residual', type=str2bool, default=True)
        parser.add_argument('--gfl_readout',type=str2bool, default=False)
        return parser


class TopoGNNModel(GCNModel):
    def __init__(self, hidden_dim, depth, num_node_features, num_classes, task,
                 lr=0.001, dropout_p=0.2, GIN=False,
                 batch_norm=False, residual=False, train_eps=True,
                 early_topo=False, residual_and_bn=False, aggregation_fn='mean',
                 dim0_out_dim=32, dim1_out_dim=32,
                 share_filtration_parameters=False, fake=False, deepset=False,
                 tanh_filtrations=False, deepset_type='full',
                 swap_bn_order=False,
                 dist_dim1=False,
                 save_filtration=False,
                 **kwargs):
        super().__init__(
            hidden_dim=hidden_dim, depth=depth,
            num_node_features=num_node_features, num_classes=num_classes,
            task=task, lr=lr, dropout_p=dropout_p, GIN=GIN,
            batch_norm=batch_norm, residual=residual, train_eps=train_eps,
            **kwargs
        )

        self.save_hyperparameters()

        self.early_topo = early_topo
        self.residual_and_bn = residual_and_bn
        self.num_filtrations = kwargs["num_filtrations"]
        self.filtration_hidden = kwargs["filtration_hidden"]
        self.num_coord_funs = kwargs["num_coord_funs"]
        self.num_coord_funs1 = self.num_coord_funs  # kwargs["num_coord_funs1"]

        self.dim1 = kwargs["dim1"]
        self.tanh_filtrations = tanh_filtrations
        self.deepset_type = deepset_type
        self.depth = depth

        self.deepset = deepset
        if self.deepset:
            self.topo1 = SetfnTopologyLayer(
                n_features=hidden_dim,
                n_filtrations=self.num_filtrations,
                mlp_hidden_dim=self.filtration_hidden,
                aggregation_fn=aggregation_fn,
                dim1=self.dim1,
                dim0_out_dim=dim0_out_dim,
                dim1_out_dim=dim1_out_dim,
                residual_and_bn=residual_and_bn,
                fake=fake,
                deepset_type=deepset_type,
                swap_bn_order=swap_bn_order,
                dist_dim1=dist_dim1
            )
        else:
            coord_funs = {"Triangle_transform": self.num_coord_funs,
                          "Gaussian_transform": self.num_coord_funs,
                          "Line_transform": self.num_coord_funs,
                          "RationalHat_transform": self.num_coord_funs
                          }

            coord_funs1 = {"Triangle_transform": self.num_coord_funs1,
                           "Gaussian_transform": self.num_coord_funs1,
                           "Line_transform": self.num_coord_funs1,
                           "RationalHat_transform": self.num_coord_funs1
                           }
            self.topo1 = CoordfnTopologyLayer(
                hidden_dim, hidden_dim,
                num_filtrations=self.num_filtrations,
                num_coord_funs=coord_funs,
                num_coord_funs1=coord_funs1,
                filtration_hidden=self.filtration_hidden,
                dim1=self.dim1,
                residual_and_bn=residual_and_bn, swap_bn_order=swap_bn_order,
                share_filtration_parameters=share_filtration_parameters, fake=fake,
                tanh_filtrations=tanh_filtrations,
                dist_dim1=dist_dim1
            )

        # number of extra dimension for each embedding from cycles (dim1)
        if self.dim1 and not dist_dim1:
            if self.deepset:
                cycles_dim = dim1_out_dim
            else:
                # classical coordinate functions
                cycles_dim = self.num_filtrations * sum(coord_funs1.values())
        else:
            cycles_dim = 0

        self.classif = torch.nn.Sequential(
            nn.Linear(hidden_dim + cycles_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, data, return_filtration=False):
        x, edge_index = data.x, data.edge_index

        x = self.embedding(x)

        if self.depth==0:
            x = x, x_dim1, filtration = self.topo1(x, data, return_filtration)
        else:
            if self.early_topo:
                # Topo layer as the second layer
                x = self.layers[0](x, edge_index=edge_index, data=data)
                x, x_dim1, filtration = self.topo1(x, data, return_filtration)
                x = F.dropout(x, p=self.hparams.dropout_p, training=self.training)
                for layer in self.layers[1:]:
                    x = layer(x, edge_index=edge_index, data=data)
            else:
                # Topo layer as the second to last layer
                for layer in self.layers[:-1]:
                    x = layer(x, edge_index=edge_index, data=data)
                x, x_dim1, filtration = self.topo1(x, data, return_filtration)
                x = F.dropout(x, p=self.hparams.dropout_p, training=self.training)
                x = self.layers[-1](x, edge_index=edge_index, data=data)

        # Pooling
        x = self.pooling_fun(x, data)

        # Aggregating the dim1 topo info if dist_dim1 == False
        if x_dim1 is not None:
            if self.task is Tasks.NODE_CLASSIFICATION:
                # Scatter graph level representation to nodes
                x_dim1 = x_dim1[data.batch]
            x_pre_class = torch.cat([x, x_dim1], axis=1)
        else:
            x_pre_class = x

        # Final classification
        x = self.classif(x_pre_class)
        if return_filtration:
            return x, filtration
        else:
            return x

    def test_step(self, batch, batch_idx):
        y = batch.y
        if self.hparams.save_filtration:
            y_hat, filtration = self(batch, return_filtration=True)
        else:
            y_hat = self(batch)
            filtration = None

        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

        self.accuracy_test(y_hat, y)

        self.log("test_acc", self.accuracy_test, on_epoch=True)

        return {"y": y, "y_hat": y_hat, "filtration": filtration}

    def test_epoch_end(self, outputs):
        y = torch.cat([output["y"] for output in outputs])
        y_hat = torch.cat([output["y_hat"] for output in outputs])

        if self.hparams.save_filtration:
            filtration = torch.nn.utils.rnn.pad_sequence(
                [output["filtration"].T for output in outputs], batch_first=True)
            if self.logger is not None:
                if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                    torch.save(filtration, os.path.join(
                        self.logger.log_dir, "filtration.pt"))
                elif isinstance(self.logger, pl.loggers.WandbLogger):
                    torch.save(filtration, os.path.join(
                        self.logger.experiment.dir, "filtration.pt"))

        y_hat_max = torch.argmax(y_hat, 1)
        if self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger):
            # Log confusion matrices
            self.logger.experiment.log({"conf_mat": wandb.plot.confusion_matrix(
                preds=y_hat_max.cpu().numpy(), y_true=y.cpu().numpy())})

    @classmethod
    def add_model_specific_args(cls, parent):
        parser = super().add_model_specific_args(parent)
        parser.add_argument('--filtration_hidden', type=int, default=24)
        parser.add_argument('--num_filtrations', type=int, default=8)
        parser.add_argument('--tanh_filtrations', type=str2bool, default=False)
        parser.add_argument('--deepset_type', type=str,
                            choices=['full', 'shallow', 'linear'], default='full')
        parser.add_argument('--swap_bn_order', type=str2bool, default=False)
        parser.add_argument('--dim1', type=str2bool, default=False)
        parser.add_argument('--num_coord_funs', type=int, default=3)
        parser.add_argument('--early_topo', type=str2bool, default=True,
                            help='Use the topo layer early in the architecture.')
        parser.add_argument('--residual_and_bn', type=str2bool, default=True,
                            help='Use residual and batch norm')
        parser.add_argument('--share_filtration_parameters', type=str2bool,
                            default=True,
                            help='Share filtration parameters of topo layer')
        parser.add_argument('--fake', type=str2bool, default=False,
                            help='Fake topological computations.')
        parser.add_argument('--deepset', type=str2bool, default=False,
                            help='Using DeepSet as coordinate function')
        parser.add_argument('--dim0_out_dim', type=int, default=32,
                            help="Inner dim of the set function of the dim0 persistent features")
        parser.add_argument('--dim1_out_dim', type=int, default=32,
                            help="Dimension of the ouput of the dim1 persistent features")
        parser.add_argument('--dist_dim1', type=str2bool, default=False)
        parser.add_argument('--aggregation_fn', type=str, default='mean')
        parser.add_argument('--save_filtration', type=str2bool, default=False)
        return parser
