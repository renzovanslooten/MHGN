import torch
from torch_scatter import scatter
from torch.nn import Sequential, ReLU, Linear, LayerNorm, GRU, LayerNorm, Dropout
from utils import *
from functools import partial
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.utils import degree, dense_to_sparse, remove_self_loops, softmax, to_dense_adj, to_dense_batch
from torch_geometric.nn.inits import glorot, zeros

#----------------------------------------------------------------------------------------------------------------------------------------------#

class BaseModel(torch.nn.Module):
    def __init__(self, is_sequence, nfeat, nhid, nout, activate_final, normalize, **kwargs):
        super(BaseModel, self).__init__()
        self.is_sequence = is_sequence
        self.kwargs = kwargs
        
        if self.is_sequence:
            self.update = GRU(nfeat, nout, num_layers=1, batch_first=True)
        else:
            if nhid is None:
                sizes = [nfeat, nout]
            elif isinstance(nhid, int):
                sizes = [nfeat, nhid, nout]
            elif isinstance(nhid, list):
                sizes = [nfeat] + nhid + [nout]
            else:
                raise ValueError

            out = []
            for i in range(len(sizes) - 1):
                out.append(Linear(sizes[i], sizes[i+1]))
                if (i < len(sizes) - 2 or activate_final):
                    out.append(ReLU(inplace=True))

            if normalize and nhid is not None:
                out.append(LayerNorm(nout))

            self.update = Sequential(*out)

  
    def forward(self, graph, concat_graph=None):
        attrs = self.collect_attrs(graph)
        if concat_graph is not None:
            concat_attrs = self.collect_attrs(concat_graph)
            attrs = [torch.cat((attr, concat_attr), dim=-1) for (attr, concat_attr) in zip(attrs, concat_attrs)]

        attrs = torch.cat(attrs, dim=-1)

        return self.update(attrs)

class EdgeModel(torch.nn.Module):
    def __init__(self, edge_project_fn, node_project_fn, activation_fn,  **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.edge_project_fn = edge_project_fn
        self.node_project_fn = node_project_fn
        self.activation_fn = activation_fn
        
        # if node_project_fn is not None:
        #     self.normalize_fn = LayerNorm(self.kwargs.get("n_out"))
        
        self.register_parameter('bias', None)
        if self.kwargs.get("edge_att_norm") == "attention":
            self.norm_att_fn = AttentionNorm(self.kwargs)
            if self.kwargs["bias"]:
                self.bias = Parameter(torch.Tensor(self.kwargs["n_heads"] * self.kwargs["n_out"] // self.kwargs["n_heads"]))
        elif self.kwargs.get("edge_att_norm") == "mean" or self.kwargs.get("edge_att_norm") == "symmetric":
            self.norm_att_fn = Normalize(self.kwargs)
        else:
            self.norm_att_fn = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_project_fn is not None:
            self.edge_project_fn.reset_parameters()
                    
        if self.node_project_fn is not None:
            self.node_project_fn.reset_parameters()
        zeros(self.bias)

    def collect_attrs(self, graphs, graphs_oth, idx):
        row, col = graphs[idx].edge_index
        n_edges = len(row)
            
        edge_attr = []
        if self.kwargs.get('with_edges_own'):
            edge_attr.append(graphs[idx].edge_attr)

        if self.kwargs.get('with_edges_below'):
            edge_attr.append(graphs[idx].edge_attr)

        if self.kwargs.get('with_edges_above'):
            edge_attr.append(edge_to_edge_broadcast(self.kwargs.get("broadcasting_meth"), graphs, idx))

        if self.kwargs.get('with_edges_depth'):
            edge_attr.append(graphs_oth[idx].edge_attr)

        if len(edge_attr) > 0:
            if self.edge_project_fn is not None:
                # Stack edges along the batch dimension
                edge_stacked = torch.cat(edge_attr, dim=0)

                # Apply the projection function
                edge_stacked_projected = self.edge_project_fn(edge_stacked)

                # Split the concatenated edges back to their original form
                edge_lengths = [edge.shape[0] for edge in edge_attr]
                edge_attr_projected = torch.split(edge_stacked_projected, edge_lengths, dim=0)

                # Concatenate corresponding edges along the feature dimension
                edge_attr = torch.cat(edge_attr_projected, dim=-1)
            else:
                edge_attr = torch.cat(edge_attr, dim = -1)
        else:
            if hasattr(graphs[idx], "edge_weight"):
                edge_weight = graphs[idx].edge_weight
            else:
                edge_weight = None

        node_attr = []
        if self.kwargs.get("with_nodes_below"):
            node_attr.append(graphs[idx].node_attr_below)

        if self.kwargs.get("with_nodes_above"): #todo: add broadcasting function
            node_attr.append(node_to_node_broadcast(self.kwargs.get("broadcast_method"), graphs, idx))

        if self.kwargs.get("with_nodes_depth"):
            if graphs_oth[idx].node_attr.dim() == 2:
                node_attr.append(graphs_oth[idx].node_attr.unsqueeze(1).repeat(1, graphs[0].node_attr.shape[1], 1))
            else:
                node_attr.append(graphs_oth[idx].node_attr[:,-1,:])

        if self.kwargs.get("with_nodes_own"):
            node_attr.append(graphs[idx].node_attr)

        if self.kwargs.get('with_globals'):
            node_attr.append(globals_to_nodes(graphs[-1].node_attr, batch = graphs[idx].batch,
                                              num_nodes = graphs[idx].batch.size(0)))

        node_attr = torch.cat(node_attr, dim = -1)

        if self.node_project_fn is not None:
            a = 1
            if isinstance(self.node_project_fn, GRU):
                # Assuming input_sequence is your input data for the GRU
                node_attr, _ = self.node_project_fn(node_attr)
                if self.norm_att_fn is not None:
                    a = self.norm_att_fn(node_attr, edge_weight, row, col, len(node_attr))
                    a = a.unsqueeze(-1).unsqueeze(-1).repeat(1, node_attr.shape[1], 1)
                    node_attr = node_attr[row].view(n_edges, self.kwargs["n_heads"], -1)
                else:
                    node_attr = node_attr[row]
                    if edge_weight is not None and self.kwargs["with_edge_weights"]:
                        a = edge_weight.unsqueeze(-1).unsqueeze(-1)
                node_attr = (node_attr * a)
            elif isinstance(self.node_project_fn, Linear):
                node_attr = self.node_project_fn(node_attr)
                if self.norm_att_fn is not None:
                    a = self.norm_att_fn(node_attr, edge_weight, row, col, len(node_attr))
                if self.kwargs.get("edge_att_norm") == "attention":
                    a = a.unsqueeze(-1)
                    node_attr = node_attr[row].view(n_edges, self.kwargs["n_heads"], -1)
                else:
                    node_attr = node_attr[row]
                    if edge_weight is not None and self.kwargs["with_edge_weights"]:
                        a = edge_weight.unsqueeze(-1)
                node_attr = (node_attr * a).flatten(1)
            else:
                raise NotImplementedError("This transformation function is not implemented")
            
            if self.bias is not None:
                node_attr = node_attr + self.bias

            if self.activation_fn is not None:
                node_attr = self.activation_fn(node_attr)
            # node_attr = self.normalize_fn(node_attr)
        else:
            a = self.norm_att_fn(node_attr, edge_weight, row, col, len(node_attr)).unsqueeze(-1)
            if node_attr.dim() == 3:
                a = a.unsqueeze(-1)
            node_attr = node_attr[row] * a

        # if graphs[idx].edge_attr is not None and graphs[idx].edge_attr.dim() == 1:
        #         node_attr = graphs[idx].edge_attr.unsqueeze(-1) * node_attr
            
        return node_attr

    def forward(self, graphs, graphs_oth, idx, concat_graphs=None, concat_graphs_oth=None):
        attrs = self.collect_attrs(graphs, graphs_oth, idx)
        if concat_graphs is not None:
            concat_attrs = self.collect_attrs(concat_graphs, concat_graphs_oth, idx)
            attrs = [torch.cat((attr, concat_attr), dim=-1) for (attr, concat_attr) in zip(attrs, concat_attrs)]
        
        graphs[idx].edge_attr = attrs
        return graphs

class NodeModel(torch.nn.Module):
    def __init__(self, node_project_fn, activation_fn, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.node_project_fn = node_project_fn
        self.activation_fn = activation_fn
        self.dropout = Dropout(self.kwargs.get("dropout_prob"))
        if self.kwargs["normalize"]:
            self.normalize_fn = torch.nn.BatchNorm1d(self.kwargs.get("n_out"))

        self.reset_parameters()

    def reset_parameters(self):                    
        if self.node_project_fn is not None:
            self.node_project_fn.reset_parameters()
        if self.kwargs["normalize"]:
            self.normalize_fn.reset_parameters()

    def collect_attrs(self, graphs, graphs_oth, idx):
        # since global node doesn't have edges
        if graphs[idx].edge_index is not None:
            row, col = graphs[idx].edge_index
            n_nodes = graphs[idx].batch.size(0)

        out = []
        if self.kwargs.get('with_receivers'):
            out.append(scatter(graphs[idx].edge_attr, col, dim=0, dim_size=n_nodes, reduce=self.kwargs.get("edge_to_node_aggr")))

        if self.kwargs.get('with_senders'):
            out.append(scatter(graphs[idx].edge_attr, row, dim=0, dim_size=n_nodes, reduce=self.kwargs.get("edge_to_node_aggr")))

        if self.kwargs.get('with_nodes_own'):
            out.append(graphs[idx].node_attr)

        if self.kwargs.get("with_nodes_below"):
            out.append(graphs[idx].node_attr_below)

        if self.kwargs.get("with_nodes_above"): #todo: add broadcasting function
            out.append(node_to_node_broadcast(self.kwargs.get("broadcast_method"), graphs, idx))

        if self.kwargs.get("with_nodes_depth"):
            if graphs_oth[idx].node_attr.dim() == 2:
                out.append(graphs_oth[idx].node_attr.unsqueeze(1).repeat(1, graphs[0].node_attr.shape[1], 1))
            else:
                out.append(graphs_oth[idx].node_attr[:,-1,:])

        if self.kwargs.get('with_globals'):
            out.append(globals_to_nodes(graphs[-1].node_attr, batch=graphs[idx].batch, num_nodes = n_nodes))

        return out

    def forward(self, graphs, graphs_oth, idx, concat_graphs=None, concat_graphs_oth=None):
        attrs = self.collect_attrs(graphs, graphs_oth, idx)
        if concat_graphs is not None:
            concat_attrs = self.collect_attrs(concat_graphs, concat_graphs_oth, idx)
            attrs = [torch.cat((attr, concat_attr), dim=-1) for (attr, concat_attr) in zip(attrs, concat_attrs)]

        attrs = torch.cat(attrs, dim=-1)
        # For HGN-GAT +
        # if len(attrs) > 1:
        #     attrs = attrs[0] + attrs[1]
        # else:
        #     attrs = torch.cat(attrs, dim=-1)
        
        if self.node_project_fn is not None:
            if isinstance(self.node_project_fn, GRU):
                node_attr, _ = self.node_project_fn(attrs)
            elif isinstance(self.node_project_fn, Linear):
                node_attr = self.node_project_fn(attrs)
        else:
            node_attr = attrs
        
        if self.activation_fn is not None:
            node_attr = self.activation_fn(node_attr)

        node_attr = self.dropout(node_attr)

        if node_attr.dim() == 3:
            node_attr = node_attr.permute(0,2,1)
            if self.kwargs["normalize"]:
                node_attr = self.normalize_fn(node_attr)
            node_attr = node_attr.permute(0,2,1)
        else:
            if self.kwargs["normalize"]:
                node_attr = self.normalize_fn(node_attr)

        graphs[idx].node_attr = node_attr
        return graphs


class SelectionModel(torch.nn.Module):
    def __init__(self, project_fn, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.selection_fn = project_fn
        
        self.reset_parameters()

    def reset_parameters(self):                    
        if self.selection_fn is not None:
            self.selection_fn.reset_parameters()

    def collect_attrs(self, graphs, graphs_oth, idx):
        out = []
        if self.kwargs.get('with_own_senders'):
            out.append(graphs[idx].node_attr)

        if self.kwargs.get('with_above_receivers'):
            out.append(node_to_node_broadcast("reverse_cluster", graphs, idx))

        if self.kwargs.get('with_selection_previous'):
            return

        if self.kwargs.get('with_selection_depth'):
            out = [graphs_oth[idx].selection]
        
        return torch.cat(out, dim=-1)

    def edge_pooling(self, graphs, idx):
        # get indices of sender and receiver nodes
        row, col = graphs[idx].edge_index

        # tmp = F.softmax(graphs[idx].selection, dim=-1)
        # assign_edge_sender = (tmp[row]).unsqueeze(-1) 
        # assign_edge_receiver = (tmp[col]).unsqueeze(1)

        # get assignments of sender and receiver nodes
        assign_edge_sender = (graphs[idx].selection[row]).unsqueeze(-1) 
        assign_edge_receiver = (graphs[idx].selection[col]).unsqueeze(1)

        # determines how well connected clusters nodes are in original graph
        assign_edge = assign_edge_sender * assign_edge_receiver

        # weighted by the original edge weights
        if self.kwargs["with_edge_weights"] and "edge_weight" in graphs[idx]:
            assign_edge = assign_edge * graphs[idx].edge_weight.unsqueeze(-1).unsqueeze(-1)

        # aggregate edges bases on connectivity of cluster nodes
        out = scatter(assign_edge, graphs[idx].batch[row], dim=0, reduce=self.kwargs["edge_to_edge_aggr"])
        
        return out
    
    def node_pooling(self, graphs, idx):
        # multiply original nodes with their assignment values (creates weighted attributed assignment edges)
        if graphs[idx].node_attr.dim() == 2:
            assign_node_attr = torch.einsum("ab, ac -> abc", graphs[idx].selection, graphs[idx].node_attr)
        else:
            assign_node_attr = torch.einsum("ab, acd -> abcd", graphs[idx].selection, graphs[idx].node_attr)

        # aggregate all incoming edges for each cluster node per batch
        cluster_node_attr = scatter(assign_node_attr, graphs[idx].batch, dim=0, reduce=self.kwargs["node_to_node_aggr"])
        
        cluster_node_attr = cluster_node_attr.flatten(end_dim=1)

        return cluster_node_attr


    def forward(self, graphs, graphs_oth, idx, concat_graphs=None, concat_graph_oth=None):
        attrs = self.collect_attrs(graphs, graphs_oth, idx)

        if self.selection_fn:
            if type(self.selection_fn).__name__ == "GraphNetwork":
                return_graphs = self.selection_fn(graphs, graphs_oth, idx)
                return_attr = F.softmax(return_graphs[idx].node_attr, dim=-1)
                graphs[idx].selection = return_attr
            else:
                graphs[idx].selection = F.softmax(self.selection_fn(attrs), dim=-1)

            s, _ = to_dense_batch(graphs[idx].selection, graphs[idx].batch)
            #     # print("idx", idx, "selection:", graphs[idx].selection.argmax(dim=1).bincount())

            dense_adj = self.edge_pooling(graphs, idx)

            if self.kwargs["normalize_adj"]:
                ind = torch.arange(dense_adj.size(-1), device=dense_adj.device)
                if self.kwargs["remove_self_loops"]:
                    dense_adj[:, ind, ind] = 0
                d = torch.einsum('ijk->ij', dense_adj)
                d = torch.sqrt(d)[:, None] + 1e-15
                dense_adj = (dense_adj / d) / d.transpose(1, 2)

            mincut_num = torch.einsum('ijj->i', dense_adj)
            d_flat = torch.einsum('ijk->ij', to_dense_adj(graphs[idx].edge_index, graphs[idx].batch))
            eye = torch.eye(d_flat.size(1)).type_as(d_flat)
            d = eye * d_flat.unsqueeze(2).expand(*d_flat.size(), d_flat.size(1)) 
            mincut_den = torch.einsum('ijj->i', torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
            mincut_loss = -(mincut_num / mincut_den)
            mincut_loss = torch.mean(mincut_loss)

            # Orthogonality regularization.
            ss = torch.matmul(s.transpose(1, 2), s)
            i_s = torch.eye(ss.size(-1)).type_as(ss)
            ortho_loss = torch.norm(
                ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
                i_s / torch.norm(i_s), dim=(-1, -2))
            ortho_loss = torch.mean(ortho_loss)
            
            edge_index, edge_weight  = dense_to_sparse(dense_adj)

            graphs[idx+1].edge_index = edge_index
            graphs[idx+1].edge_weight = edge_weight

            graphs[idx].loss_1 += mincut_loss
            graphs[idx].loss_2 += ortho_loss
            graphs[idx+1].batch = torch.arange(dense_adj.shape[0]).repeat_interleave(dense_adj.shape[1]).to(dense_adj.device)
        else:
            if self.kwargs["n_out"] == 1:
                graphs[idx].selection = torch.ones(graphs[idx].size(0), 1).to(graphs[idx].batch.device)
            else:
                if self.kwargs["with_selection_previous"]:
                    graphs_oth = graphs
                graphs[idx].selection = graphs_oth[idx].selection

                graphs[idx+1].edge_index = graphs_oth[idx+1].edge_index
                graphs[idx+1].edge_weight = graphs_oth[idx+1].edge_weight
                graphs[idx+1].batch = graphs_oth[idx+1].batch
            
        graphs[idx+1].node_attr_below = self.node_pooling(graphs, idx)
        
        graphs[idx+1].ptr = torch.cumsum(torch.tensor([0] + [self.kwargs["n_out"]] * (len(graphs[idx].ptr)-1)), dim=0).to("cuda:0")
        
        return graphs


#-------------------------------------------------------------------------------------------------------------------------------#


class Normalize(torch.nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        if self.kwargs.get("edge_att_norm") == "mean":
            self.normalizer = partial(self.normalizer_mean)
        elif self.kwargs.get("edge_att_norm") == "symmetric":
            self.normalizer = partial(self.normalizer_GCN)

    def normalizer_mean(self, row, col, num_nodes):
        deg = degree(row, num_nodes)
        return (1 / deg)[row]

    def normalizer_GCN(self, row, col, num_nodes):
        deg = degree(col)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * deg_inv_sqrt[col]

    def forward(self, node_attr, edge_weight, row, col, num_nodes):
        norm = self.normalizer(row, col, num_nodes)

        if self.kwargs["with_edge_weights"]:
            if edge_weight is not None and edge_weight.dim() == 1:
                norm += edge_weight

        return norm



#-------------------------------------------------------------------------------------------------------------------------------#

class AttentionNorm(torch.nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.n_out = kwargs["n_out"] // kwargs["n_heads"]

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, self.kwargs["n_heads"], self.n_out))
        self.att_dst = Parameter(torch.Tensor(1, self.kwargs["n_heads"], self.n_out))
        if self.kwargs["with_edge_weights"]:
            self.att_edge = Parameter(torch.Tensor(1, self.kwargs["n_heads"]))
        else:
            self.register_parameter('att_edge', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)

    def forward(self, node_attr, edge_weight, row, col, num_nodes):
        H, C = self.kwargs["n_heads"], self.n_out

        node_attr = node_attr.view(-1, H, C)

        x_src = node_attr[row]
        x_dst = node_attr[col]

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst

        if self.att_edge is not None:
            if edge_weight is not None and edge_weight.dim() == 1:
                alpha += (self.att_edge * edge_weight.unsqueeze(-1))

        alpha = F.leaky_relu(alpha, 0.2)
        # tested and column is good
        alpha = softmax(alpha, col)
    
        return alpha

#-------------------------------------------------------------------------------------------------------------------------------#


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1).pow(0.5))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum("abc,cd->abd", (input, self.weight))
        output = torch.einsum("abc,abd->abd", (adj, support))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'