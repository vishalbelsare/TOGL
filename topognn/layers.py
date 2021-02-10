"""Implementation of layers following Benchmarking GNNs paper."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter
from torch_persistent_homology.persistent_homology_cpu import (
    compute_persistence_homology_batched_mt,
)

import topognn.coord_transforms as coord_transforms
from topognn.data_utils import remove_duplicate_edges


class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, dropout, batch_norm, residual=True
    ):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
        self.conv = GCNConv(in_features, out_features, add_self_loops=False)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GINLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        dropout,
        batch_norm,
        mlp_hidden_dim=None,
        residual=True,
        train_eps=False
    ):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = in_features

        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
        gin_net = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        self.conv = GINConv(gin_net, train_eps=train_eps)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        # Apply aggregation function over graph
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class DeepSetLayerDim1(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn

    def forward(self, x, edge_slices, mask=None):
        '''
        Mask is True where the persistence (x) is observed.
        '''
        # Apply aggregation function over graph

        # Computing the equivalent of batch over edges.
        edge_diff_slices = (edge_slices[1:]-edge_slices[:-1]).to(x.device)
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(torch.arange(
            n_batch, device=x.device), edge_diff_slices)
        # Only aggregate over edges with non zero persistence pairs.
        if mask is not None:
            batch_e = batch_e[mask]

        xm = scatter(x, batch_e, dim=0,
                     reduce=self.aggregation_fn, dim_size=n_batch)

        xm = self.Lambda(xm)

        # xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        # xm = self.Lambda(xm)
        # x = self.Gamma(x)
        # x = x - xm[batch, :]
        return xm


def fake_persistence_computation(filtered_v_, edge_index, vertex_slices, edge_slices, batch):
    device = filtered_v_.device
    num_filtrations = filtered_v_.shape[1]
    filtered_e_, _ = torch.max(torch.stack(
        (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)

    # Make fake tuples for dim 0
    persistence0_new = filtered_v_.unsqueeze(-1).expand(-1, -1, 2)

    edge_slices = edge_slices.to(device)
    bs = edge_slices.shape[0] - 1
    # Make fake dim1 with unpaired values
    # unpaired_values = scatter(filtered_v_, batch, dim=0, reduce='max')
    unpaired_values = torch.zeros((bs, num_filtrations), device=device)
    persistence1_new = torch.zeros(
        edge_index.shape[1], filtered_v_.shape[1], 2, device=device)

    n_edges = edge_slices[1:] - edge_slices[:-1]
    random_edges = (
        edge_slices[0:-1].unsqueeze(-1) +
        torch.floor(
            torch.rand(size=(bs, num_filtrations), device=device)
            * n_edges.float().unsqueeze(-1)
        )
    ).long()

    persistence1_new[random_edges, torch.arange(num_filtrations).unsqueeze(0), :] = (
        torch.stack([
            unpaired_values,
            filtered_e_[
                    random_edges, torch.arange(num_filtrations).unsqueeze(0)]
        ], -1)
    )
    return persistence0_new.permute(1, 0, 2), persistence1_new.permute(1, 0, 2), None


class CoordfnTopologyLayer(nn.Module):
    """Topological layer using coordinate functions."""

    def __init__(self, features_in, features_out, num_filtrations,
                 num_coord_funs, filtration_hidden, num_coord_funs1=None,
                 dim1=False, residual_and_bn=False,
                 share_filtration_parameters=False, fake=False,
                 tanh_filtrations=False, swap_bn_order=False, dist_dim1=False):
        """
        num_coord_funs is a dictionary with the numbers of coordinate functions of each type.
        dim1 is a boolean. True if we have to return dim1 persistence.
        """
        super().__init__()

        self.dim1 = dim1

        self.features_in = features_in
        self.features_out = features_out

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.filtration_hidden = filtration_hidden
        self.residual_and_bn = residual_and_bn
        self.share_filtration_parameters = share_filtration_parameters
        self.fake = fake
        self.swap_bn_order = swap_bn_order
        self.dist_dim1 = dist_dim1

        self.total_num_coord_funs = sum(num_coord_funs.values())

        self.coord_fun_modules = torch.nn.ModuleList([
            getattr(coord_transforms, key)(output_dim=num_coord_funs[key])
            for key in num_coord_funs
        ])

        if self.dim1:
            assert num_coord_funs1 is not None
            self.coord_fun_modules1 = torch.nn.ModuleList([
                getattr(coord_transforms, key)(output_dim=num_coord_funs1[key])
                for key in num_coord_funs1
            ])

        final_filtration_activation = nn.Tanh() if tanh_filtrations else nn.Identity()
        if self.share_filtration_parameters:
            self.filtration_modules = torch.nn.Sequential(
                torch.nn.Linear(self.features_in, self.filtration_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filtration_hidden, num_filtrations),
                final_filtration_activation
            )
        else:
            self.filtration_modules = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.features_in, self.filtration_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.filtration_hidden, 1),
                    final_filtration_activation
                ) for _ in range(num_filtrations)
            ])

        if self.residual_and_bn:
            in_out_dim = self.num_filtrations * self.total_num_coord_funs
            features_out = features_in
            self.bn = nn.BatchNorm1d(features_out)
            if self.dist_dim1 and self.dim1:
                self.out1 = torch.nn.Linear(
                    self.num_filtrations * self.total_num_coord_funs, features_out)
        else:
            if self.dist_dim1:
                in_out_dim = self.features_in + 2 * \
                    self.num_filtrations * self.total_num_coord_funs
            else:
                in_out_dim = self.features_in + self.num_filtrations * self.total_num_coord_funs

        self.out = torch.nn.Linear(in_out_dim, features_out)

    def compute_persistence(self, x, batch, return_filtration=False):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        edge_index = batch.edge_index
        if self.share_filtration_parameters:
            filtered_v_ = self.filtration_modules(x)
        else:
            filtered_v_ = torch.cat([filtration_mod.forward(x)
                                     for filtration_mod in self.filtration_modules], 1)
        filtered_e_, _ = torch.max(torch.stack(
            (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)

        vertex_slices = torch.Tensor(batch.__slices__['x']).long()
        edge_slices = torch.Tensor(batch.__slices__['edge_index']).long()

        if self.fake:
            return fake_persistence_computation(
                filtered_v_, edge_index, vertex_slices, edge_slices, batch.batch)

        vertex_slices = vertex_slices.cpu()
        edge_slices = edge_slices.cpu()

        filtered_v_ = filtered_v_.cpu().transpose(1, 0).contiguous()
        filtered_e_ = filtered_e_.cpu().transpose(1, 0).contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(
            filtered_v_, filtered_e_, edge_index,
            vertex_slices, edge_slices)
        persistence0_new = persistence0_new.to(x.device)
        persistence1_new = persistence1_new.to(x.device)

        if return_filtration:
            return persistence0_new, persistence1_new, filtered_v_
        else:
            return persistence0_new, persistence1_new, None

    def compute_coord_fun(self, persistence, batch, dim1=False):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """
        if dim1:
            coord_activation = torch.cat(
                [mod.forward(persistence) for mod in self.coord_fun_modules1], 1)
        else:
            coord_activation = torch.cat(
                [mod.forward(persistence) for mod in self.coord_fun_modules], 1)

        return coord_activation

    def compute_coord_activations(self, persistences, batch, dim1=False):
        """
        Return the coordinate functions activations pooled by graph.
        Output dims : list of length number of filtrations with elements : [N_graphs in batch, number fo coordinate functions]
        """

        coord_activations = [self.compute_coord_fun(
            persistence, batch=batch, dim1=dim1) for persistence in persistences]
        return torch.cat(coord_activations, 1)

    def collapse_dim1(self, activations, mask, slices):
        """
        Takes a flattened tensor of activations along with a mask and collapses it (sum) to have a graph-wise features

        Inputs : 
        * activations [N_edges,d]
        * mask [N_edge]
        * slices [N_graphs]
        Output:
        * collapsed activations [N_graphs,d]
        """
        collapsed_activations = []
        for el in range(len(slices)-1):
            activations_el_ = activations[slices[el]:slices[el+1]]
            mask_el = mask[slices[el]:slices[el+1]]
            activations_el = activations_el_[mask_el].sum(axis=0)
            collapsed_activations.append(activations_el)

        return torch.stack(collapsed_activations)

    def forward(self, x, batch, return_filtration=False):
        # Remove the duplicate edges.
        batch = remove_duplicate_edges(batch)

        persistences0, persistences1, filtration = self.compute_persistence(
            x, batch, return_filtration)

        coord_activations = self.compute_coord_activations(
            persistences0, batch)
        if self.dim1:
            persistence1_mask = (persistences1 != 0).any(2).any(0)
            # TODO potential save here by only computing the activation on the masked persistences
            coord_activations1 = self.compute_coord_activations(
                persistences1, batch, dim1=True)
            # Below returns a vector for each graph
            graph_activations1 = self.collapse_dim1(
                coord_activations1, persistence1_mask, batch.__slices__["edge_index"])
        else:
            graph_activations1 = None

        if self.residual_and_bn:
            out_activations = self.out(coord_activations)

            if self.dim1 and self.dist_dim1:
                out_activations += self.out1(graph_activations1)[batch]
                graph_activations1 = None
            if self.swap_bn_order:
                out_activations = self.bn(out_activations)
                out_activations = x + F.relu(out_activations)
            else:
                out_activations = self.bn(out_activations)
                out_activations = x + out_activations
        else:
            concat_activations = torch.cat((x, coord_activations), 1)
            out_activations = self.out(concat_activations)
            out_activations = F.relu(out_activations)

        return out_activations, graph_activations1, filtration


class SetfnTopologyLayer(nn.Module):

    def __init__(self, n_features, n_filtrations, mlp_hidden_dim,
                 aggregation_fn, dim0_out_dim, dim1_out_dim, dim1,
                 residual_and_bn, fake, deepset_type='full',
                 swap_bn_order=False, dist_dim1=False):
        super().__init__()
        self.filtrations = nn.Sequential(
            nn.Linear(n_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_filtrations),
        )

        assert deepset_type in ['linear', 'shallow', 'full']

        self.num_filtrations = n_filtrations
        self.residual_and_bn = residual_and_bn
        self.swap_bn_order = swap_bn_order
        self.dist_dim1 = dist_dim1

        self.dim1_flag = dim1
        if self.dim1_flag:
            self.set_fn1 = nn.ModuleList([
                nn.Linear(n_filtrations * 2, dim1_out_dim),
                nn.ReLU(),
                DeepSetLayerDim1(
                    in_dim=dim1_out_dim, out_dim=n_features if residual_and_bn and dist_dim1 else dim1_out_dim, aggregation_fn=aggregation_fn),
            ])

        if deepset_type == 'linear':
            self.set_fn0 = nn.ModuleList([nn.Linear(
                n_filtrations * 2,
                n_features if residual_and_bn else dim0_out_dim, aggregation_fn)
            ])
        elif deepset_type == 'shallow':
            self.set_fn0 = nn.ModuleList(
                [
                    nn.Linear(n_filtrations * 2, dim0_out_dim),
                    nn.ReLU(),
                    DeepSetLayer(
                        dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn),
                ]
            )
        else:
            self.set_fn0 = nn.ModuleList(
                [
                    nn.Linear(n_filtrations * 2, dim0_out_dim),
                    nn.ReLU(),
                    DeepSetLayer(dim0_out_dim, dim0_out_dim, aggregation_fn),
                    nn.ReLU(),
                    DeepSetLayer(
                        dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn),
                ]
            )
        if residual_and_bn:
            self.bn = nn.BatchNorm1d(n_features)
        else:
            if dist_dim1:
                self.out = nn.Sequential(
                    nn.Linear(dim0_out_dim + dim1_out_dim +
                              n_features, n_features),
                    nn.ReLU()
                )
            else:
                self.out = nn.Sequential(
                    nn.Linear(dim0_out_dim + n_features, n_features),
                    nn.ReLU()
                )
        self.fake = fake

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices, batch, return_filtration=False):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        filtered_v = self.filtrations(x)
        if self.fake:
            return fake_persistence_computation(
                filtered_v, edge_index, vertex_slices, edge_slices, batch)

        filtered_e, _ = torch.max(
            torch.stack((filtered_v[edge_index[0]], filtered_v[edge_index[1]])), axis=0
        )

        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(
            filtered_v, filtered_e, edge_index, vertex_slices, edge_slices
        )
        persistence0 = persistence0_new.to(x.device)
        persistence1 = persistence1_new.to(x.device)

        if return_filtration:
            return persistence0, persistence1, filtered_v
        else:
            return persistence0, persistence1, None

    def forward(self, x, data, return_filtration):

        # Remove the duplucate edges
        data = remove_duplicate_edges(data)

        edge_index = data.edge_index
        vertex_slices = torch.Tensor(data.__slices__['x']).cpu().long()
        edge_slices = torch.Tensor(data.__slices__['edge_index']).cpu().long()
        batch = data.batch

        pers0, pers1, filtration = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices, batch, return_filtration
        )

        x0 = pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)

        for layer in self.set_fn0:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        if self.dim1_flag:
            # Dim 1 computations.
            pers1_reshaped = pers1.permute(1, 0, 2).reshape(pers1.shape[1], -1)
            pers1_mask = ~((pers1_reshaped == 0).all(-1))
            x1 = pers1_reshaped[pers1_mask]
            for layer in self.set_fn1:
                if isinstance(layer, DeepSetLayerDim1):
                    x1 = layer(x1, edge_slices, mask=pers1_mask)
                else:
                    x1 = layer(x1)
        else:
            x1 = None

        # Collect valid
        # valid_0 = (pers1 != 0).all(-1)

        if self.residual_and_bn:
            if self.dist_dim1 and self.dim1_flag:
                x0 = x0 + x1[batch]
                x1 = None
            if self.swap_bn_order:
                x = x + F.relu(self.bn(x0))
            else:
                x = x + self.bn(F.relu(x0))
        else:
            if self.dist_dim1 and self.dim1_flag:
                x0 = torch.cat([x0, x1[batch]], dim=-1)
                x1 = None
            x = self.out(torch.cat([x, x0], dim=-1))

        return x, x1, filtration
