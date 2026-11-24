# Added for CODA (2026): GAT backbone in the style of the GOOD GCN. Part of the GPL-3.0 GOOD subset in this directory.
r"""
The Graph Attention Network from the `"Graph Attention Networks"
<https://arxiv.org/abs/1710.10903>`_ paper.
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, Size
from torch_sparse import SparseTensor

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .MolEncoders import AtomEncoder


@register.model_register
class GAT(GNNBasic):
    r"""
    The Graph Attention Network from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args
            (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`,
             :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GATFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GAT model implementation.

        Args:
            *args (list): argument list for the use of arguments_read.
            **kwargs (dict): key word arguments for the use of arguments_read.

        Returns (Tensor):
            label predictions
        """
        out_readout = self.feat_encoder(*args, **kwargs)
        out = self.classifier(out_readout)
        return out


class GATFeatExtractor(GNNBasic):
    r"""
    GAT feature extractor using the :class:`~GATEncoder`.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args
            (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`,
             :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GATFeatExtractor, self).__init__(config)
        self.encoder = GATEncoder(config, **kwargs)
        self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GAT feature extractor using the :class:`~GATEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read.
            **kwargs (dict): key word arguments for the use of arguments_read.

        Returns (Tensor):
            node feature representations
        """
        ret = self.arguments_read(*args, **kwargs)
        edge_weight = None
        batch_size = None
        if self.config.model.model_level == 'node':
            x, edge_index, edge_weight, batch = ret
        elif len(ret) == 5:
            x, edge_index, _, batch, batch_size = ret
        elif len(ret) == 4:
            x, edge_index, _, batch = ret
        else:
            x, edge_index, batch = ret
        # Remove keys already extracted to avoid "multiple values" error
        fwd_kwargs = {k: v for k, v in kwargs.items() if k not in ('edge_weight', 'data', 'batch_size', 'ood_algorithm')}
        out_readout = self.encoder(x, edge_index, edge_weight, batch, batch_size, **fwd_kwargs)
        return out_readout


class GATEncoder(BasicEncoder):
    r"""
    The GAT encoder using the :class:`~GATConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args
            (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`,
             :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):
        super(GATEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model.model_layer
        num_heads = getattr(config.model, 'num_heads', 4)
        self.without_readout = kwargs.get('without_readout')

        # Mol datasets use AtomEncoder to embed categorical node features
        self.atom_encoder = None
        if getattr(config.dataset, 'dataset_type', None) == 'mol' and not kwargs.get('without_embed'):
            self.atom_encoder = AtomEncoder(config.model.dim_hidden)
            in_dim = config.model.dim_hidden
        elif kwargs.get('without_embed'):
            in_dim = config.model.dim_hidden
        else:
            in_dim = config.dataset.dim_node
        head_dim = config.model.dim_hidden // num_heads
        self.conv1 = GATConv(in_dim, head_dim, heads=num_heads,
                             dropout=config.model.dropout_rate)
        self.convs = nn.ModuleList(
            [
                GATConv(config.model.dim_hidden, head_dim, heads=num_heads,
                        dropout=config.model.dropout_rate)
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_weight, batch, batch_size=None, **kwargs):
        r"""
        The GAT encoder.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_weight (Tensor): edge weights (not used in GAT, kept for API compatibility)
            batch (Tensor): batch indicator
            batch_size (int): batch size for graph-level pooling

        Returns (Tensor):
            node feature representations
        """
        if self.atom_encoder is not None:
            x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        if batch_size is not None:
            out_readout = self.readout(post_conv, batch, batch_size)
        else:
            out_readout = self.readout(post_conv, batch)
        return out_readout


class GATConv(gnn.GATConv):
    r"""
    The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    Args:
        *args (list): argument list.
        **kwargs (dict): additional key word arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.__explain__ = False
        self.__edge_mask__ = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None) -> Tensor:
        r"""
        The GAT graph attentional operator.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes (optional)
            return_attention_weights (bool): whether to return attention weights

        Returns (Tensor):
            node feature representations
        """
        out = super().forward(x, edge_index, edge_attr,
                              return_attention_weights=return_attention_weights)
        if isinstance(out, tuple):
            # When return_attention_weights is True, returns (out, (edge_index, alpha))
            return out
        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor or SparseTensor): defines the underlying
                graph connectivity/message passing flow.
            size (tuple, optional): The size of the assignment matrix.
            **kwargs: Any additional data for message construction and aggregation.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask.sigmoid()
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)
