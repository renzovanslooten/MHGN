import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import copy
import scipy.sparse as sp
from itertools import repeat
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
import os.path as osp
import os
from torch_scatter import scatter_sum
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops, remove_self_loops

#-----------------------------------------------------------------------------------------------------------------------------------------#

# class RMSELoss(torch.nn.Module):
#     def __init__(self):
#         super(RMSELoss,self).__init__()

#     def forward(self,predicted_y,y):
#         criterion = torch.nn.MSELoss()
#         loss = torch.sqrt(criterion(predicted_y, y))
#         return loss
    
#-----------------------------------------------------------------------------------------------------------------------------------------#

def loss_SBM(label, pred, n_classes, device):
    # calculating label weights for weighted loss computation
    V = label.size(0)
    label_count = torch.bincount(label)
    label_count = label_count[label_count.nonzero()].squeeze()
    cluster_sizes = torch.zeros(n_classes).long().to(device)
    cluster_sizes[torch.unique(label)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes>0).float()
    
    # weighted cross-entropy for unbalanced classes
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred, label)
    return loss

def loss_MNIST_CIFAR(label, pred, n_classes, device):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred, label)
    return loss

def accuracy_SBM(targets, scores):
    scores = torch.argmax(scores, dim=1)
    unique_classes = torch.unique(targets)
    class_accuracies = []

    for class_label in unique_classes:
        class_indices = (targets == class_label)
        class_accuracy = targets[class_indices].eq(scores[class_indices]).sum().item() / class_indices.sum()
        class_accuracies.append(class_accuracy)

    weighted_accuracy = torch.Tensor(class_accuracies).mean()
    return weighted_accuracy

def accuracy_MNIST_CIFAR(targets, scores):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc / targets.size(0)

def create_adjecency_matrix(buurt_polygons):
    idx_map = {j: i for i, j in enumerate(buurt_polygons.index)}
    
    # Create a new GeoDataFrame and assign the 'polygon' column from buurt_polygons to its 'geometry' column
    buurt_gdf = gpd.GeoDataFrame()
    buurt_gdf['geometry'] = buurt_polygons['polygon']

    edges_unordered = np.concatenate([list(zip(buurt_gdf.index[buurt_gdf.touches(row.geometry)], repeat(code))) 
                                for code, row in buurt_gdf.iterrows()])
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(buurt_gdf), len(buurt_gdf)))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def count_parameters(model, model_name, print_name, print_dim):
    counter = 0
    for name, param in model.named_parameters():
        param_dim = param.data.size()
        if print_name and print_dim:
            print(name, ": ", param_dim)
        elif print_dim:
            print(param_dim)
        counter += param_dim.numel()
    print("\n Total number of Parameters ", model_name, ": ", counter)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def create_wfv_splits(target_df, test_idx):
    if test_idx < 1 | test_idx > 6: print("the test_idx range should be between 1 and 7")
    first_year_length = target_df.groupby(pd.Grouper(freq="Y")).size()[0]
    month_lengths = target_df.iloc[first_year_length:].groupby(pd.Grouper(freq="M")).size().values
    month_lengths = month_lengths[::2] + month_lengths[1::2]
    month_idx = first_year_length + np.hstack([0, np.cumsum(month_lengths)])
    train_ranges = [range(x) for x in month_idx][-test_idx-1:-1]
    val_ranges = [range(x, y) for x, y in list(zip(month_idx,month_idx[1:]))][-test_idx:]
    wfv_splits = np.array(list(zip(train_ranges,val_ranges)), dtype=object)
    train_size = [target_df.iloc[x].notnull().sum().sum() for x, _ in wfv_splits]
    test_size = [target_df.iloc[y].notnull().sum().sum() for _, y in wfv_splits]
    return wfv_splits, train_size, test_size

def create_path(model_root, input_types, dataset):
    model_dir = osp.join(model_root, "_".join(sorted(input_types)))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    config_values = "_".join(sum([[key + ":" + str(value)] for key, value in dataset.config.items() 
                                    if key not in ["nfeat", "seq_len", "num_nodes"]], []))
    model_dir = osp.join(model_dir, config_values)                              
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)                          
    return model_dir

def create_cluster_adjecency_matrix(kmax, sil_metric):
    bbga_features = pd.read_pickle("../../data/processed_inputs/bbga_pca.pkl")
    X = bbga_features.loc[2017].T.values
    
    sil = []
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric = sil_metric))
    best_n_clusters = np.argmax(sil) + 2
    kmeans = KMeans(n_clusters = best_n_clusters).fit(X)
    labels = kmeans.labels_

    edges = np.stack([[i, j] for i in range (len(np.unique(labels)))
            for j, a in enumerate(labels) if a == i])

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(bbga_features.shape[1]), len(bbga_features.shape[1])))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj

def get_edge_counts(edge_index, batch):
    return torch.bincount(batch[edge_index[0, :]])


def globals_to_nodes(global_attr, batch=None, num_nodes=None):
    if batch is not None:
        _, counts = torch.unique(batch, return_counts=True)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0)
                                        for idx, rep in enumerate(counts)], dim=0)
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_nodes is not None, "number of nodes should be specified."
        casted_global_attr = torch.cat([global_attr] * num_nodes, dim=0)
    return casted_global_attr


def globals_to_edges(global_attr, edge_index=None, batch=None, num_edges=None):
    if batch is not None:
        assert edge_index is not None, "edge index should be specified"
        edge_counts = get_edge_counts(edge_index, batch)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0)
                                        for idx, rep in enumerate(edge_counts)], dim=0)
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_edges is not None, "number of edges should be specified"
        casted_global_attr = torch.cat([global_attr] * num_edges, dim=0)
    return casted_global_attr


def edges_to_globals(edge_attr, edge_index=None, batch=None, num_edges=None, num_globals=None):
    if batch is None:
        edge_attr_aggr = torch.sum(edge_attr, dim=0, keepdim=True)
    else:
        node_indices = torch.unique(batch)
        edge_counts = get_edge_counts(edge_index, batch)
        assert sum(edge_counts) == num_edges
        # indices = [idx.view(1, 1) for idx, count in zip(node_indices, edge_counts) for _ in range(count)]
        indices = [torch.repeat_interleave(idx, count) for idx, count in zip(node_indices, edge_counts)]
        indices = torch.cat(indices)
        edge_attr_aggr = scatter_sum(edge_attr, index=indices, dim=0, dim_size=num_globals)
    return edge_attr_aggr


def edge_to_edge_broadcast(broadcast_method, graphs, idx):
    if broadcast_method == "reverse_cluster":
        graphs[idx+1].edge_attr
    raise NotImplementedError("This function is not implemented yet.")


# def node_to_node_broadcast(broadcast_method, graphs, idx):
#     batch_size = len(graphs[idx]["ptr"]) -1
#     s_dim = graphs[idx].selection.shape
#     v_dim = graphs[idx+1].node_attr.shape
#     s_dim = [batch_size] + [int(s_dim[0] / batch_size)] + list(s_dim[1:])
#     v_dim = [batch_size] + [int(v_dim[0] / batch_size)] + list(v_dim[1:])
#     if len(v_dim) == 3:
#         return torch.einsum("abc, acd -> abd", graphs[idx].selection.reshape(s_dim), graphs[idx+1].node_attr.reshape(v_dim)).flatten(end_dim=1)
#     else:
#         return torch.einsum("abc, acde -> abde", graphs[idx].selection.reshape(s_dim), graphs[idx+1].node_attr.reshape(v_dim)).flatten(end_dim=1)
        
def node_to_node_broadcast(broadcast_method, graphs, idx):
    nodes_per_graph = graphs[idx].ptr[1:] - graphs[idx].ptr[:-1]

    broadcasted_attrs = torch.repeat_interleave(
        graphs[idx+1].node_attr.view(nodes_per_graph.shape[0], -1, *graphs[idx+1].node_attr.shape[1:]),
        nodes_per_graph, dim=0)
    
    if broadcasted_attrs.dim() == 3:
        weighted_broadcasted_attr = broadcasted_attrs * graphs[idx].selection.unsqueeze(-1)
    else:
        # embedding blocks don't have selection yet, but since it only the case for the global node, we can catch it
        if "selection" in graphs[idx]:
            weighted_broadcasted_attr = broadcasted_attrs * graphs[idx].selection.unsqueeze(-1).unsqueeze(-1)
        else:
            weighted_broadcasted_attr = broadcasted_attrs

    return torch.sum(weighted_broadcasted_attr, dim=1)

def nodes_to_globals(node_attr, batch=None, num_globals=None):
    if batch is None:
        x_aggr = torch.sum(node_attr, dim=0, keepdim=True)
    else:
        x_aggr = scatter_sum(node_attr, index=batch, dim=0, dim_size=num_globals)
    return x_aggr


# def edges_to_nodes(edge_attr, indices, num_nodes=None):
#     edge_attr_aggr = scatter_sum(edge_attr, indices, dim=0, dim_size=num_nodes)
#     return edge_attr_aggr


def nodes_to_clusters(self, v, edge_index, batch):
    S = F.softmax(self.soft_matrix(v), dim=-1)
    S = S.view(-1, self.n_nodes, S.shape[-1]).permute(0,2,1)
    v = v.view(-1, self.n_nodes, v.shape[-1])
    cluster_attrs = torch.bmm(S, v)
    return cluster_attrs, S.flatten(0,1)

def dense_to_sparse_with_attr(adj):
    adj2 = adj.abs().sum(dim=-1)  
    index = adj2.nonzero(as_tuple=True)
    edge_attr = adj[index]
    batch = index[0] * adj.size(-2)
    index = (batch + index[1], batch + index[2])
    edge_index = torch.stack(index, dim=0)
    return edge_index, edge_attr

def not_all_nodes_have_self_loops(batch):
    edge_index = batch.edge_index
    row, col = edge_index
    mask = row == col
    loop_nodes = row[mask].unique()
    num_nodes = batch.num_nodes
    return loop_nodes.numel() != num_nodes

            
def hierarchical_graphs(n_cluster_nodes, batch, self_loops):
    n_graphs = len(n_cluster_nodes) + 1

    if self_loops and not_all_nodes_have_self_loops(batch):
        edge_index, edge_attr = remove_self_loops(
            batch.edge_index, batch.edge_attr)
        batch.edge_index, batch.edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value="mean")

    ST_graphs = []
    if hasattr(batch, "v_st"):
        ST_graphs.append(Batch(batch = getattr(batch, "batch"), edge_index = getattr(batch, "edge_index"), 
                            node_attr = getattr(batch, "v_st"), ptr = getattr(batch, "ptr"), loss_1 = 0, loss_2 = 0))
        del batch.v_st
    else:
        ST_graphs.append(Batch(loss_1 = 0, loss_2 = 0))

    if hasattr(batch, "edge_attr"):
        batch.edge_weight = batch.edge_attr
        ST_graphs[0].edge_weight = batch.edge_attr
        del batch.edge_attr

    for _ in range(n_graphs -2):
        ST_graphs.append(Batch(loss_1 = 0, loss_2 = 0))
    if n_graphs > 1:
        if hasattr(batch, "g_t") and n_cluster_nodes[-1] == 1:    
            ST_graphs.append(Batch(batch = torch.arange(batch.batch.max() + 1), node_attr = getattr(batch, "g_t"),
                                ptr = torch.arange(batch.batch.max() + 2), loss_1 = 0, loss_2 = 0))
            del batch.g_t
        else:
            ST_graphs.append(Batch(loss_1 = 0, loss_2 = 0))

    S_graphs = [Batch()]
    for _ in range(n_graphs -2):
        S_graphs.append(Batch(loss_1 = 0, loss_2 = 0))
    if n_graphs > 1:
        if hasattr(batch, "g_s") and n_cluster_nodes[-1] == 1:
            S_graphs.append(Batch(batch = torch.arange(batch.batch.max() + 1), node_attr = getattr(batch, "g_s"),
                                ptr = torch.arange(batch.batch.max() + 2), loss_1 = 0, loss_2 = 0))
            del batch.g_s
        else:
            S_graphs.append(Batch(loss_1 = 0, loss_2 = 0))

    if hasattr(batch, 'v_s'):
        batch.node_attr = batch.v_s
        del batch.v_s

    S_graphs[0] = batch

    for graph in S_graphs:
        graph.loss_1 = 0
        graph.loss_2 = 0

    return S_graphs, ST_graphs


_DEFAULT_EDGE_BLOCK_OPT = {
"bias": True,
"with_edges_own": False,
"with_edges_below": False,
"with_edges_above": False,
"with_edges_depth": False,
"with_globals": False,
"with_nodes_below": False,
"with_nodes_above": False,
"with_nodes_depth": False,
"with_nodes_own": False,
"edge_att_norm": None,
}

_DEFAULT_NODE_BLOCK_OPT = {
    "bias": True,
    "with_receivers": True,
    "with_nodes_below": False,
    "with_nodes_own": False,
    "with_nodes_above": False,
    "with_nodes_depth": False,
    "with_globals": False
}

_DEFAULT_SELECT_BLOCK_OPT = {
    "bias": True,
    "with_selection_depth": False,
    "with_selection_previous": False,
    "with_own_senders": False,
    "remove_self_loops": False
}

def build_network_params(model_dict, pooling_dict, data_dict, layer_dict, setup_dict, n_cluster_nodes):
    n_graph = len(n_cluster_nodes) + 1
    n_hid_layer = setup_dict["n_hid_layer"]

    model_dict = merge_model_parameters(pooling_dict, model_dict)
    
    input_feats_dict = n_features_graphs(data_dict, n_cluster_nodes)
    setup_dict = {**data_dict, **setup_dict, **input_feats_dict}

    params = {}
    if sum(setup_dict["n_edge_feat_S"]) + sum(setup_dict["n_node_feat_S"]) > 0:
        params["params_S"] = {}
        if setup_dict["HMGN_emb"]:
            params["params_S"] = {"enc_params": [block_params(model_dict, setup_dict, layer_dict["emb_params"], idx, 
                                                              is_ST = False, is_enc = True) for idx in range(n_graph)]}
        if n_hid_layer > 0:
            params["params_S"]["pro_params"] = [block_params(model_dict, setup_dict, layer_dict["pro_params"], idx, 
                                                             is_ST = False) for idx in range(n_graph)]
        if setup_dict["HMGN_dec"]:
            params["params_S"]["dec_params"] = [block_params(model_dict, setup_dict, layer_dict["dec_params"], idx, 
                                                         is_ST = False, is_dec = True) for idx in range(n_graph)]

    if sum(setup_dict["n_edge_feat_ST"]) + sum(setup_dict["n_node_feat_ST"]) > 0:
        params["params_ST"] = {}
        if setup_dict["HMGN_emb"]:
            params["params_ST"] = {"enc_params": [block_params(model_dict, setup_dict, layer_dict["emb_params"], idx,
                                                               is_ST = True, is_enc = True) for idx in range(n_graph)]}
        if n_hid_layer > 0:
            params["params_ST"]["pro_params"] = [block_params(model_dict, setup_dict, layer_dict["pro_params"], idx, 
                                                              is_ST = True) for idx in range(n_graph)]
    return params

def block_params(model_dict, setup_dict, layer_dict, idx, is_ST, is_enc=False, is_dec=False):
    
    n_node_feat = setup_dict["n_node_feat_ST"] if is_ST else setup_dict["n_node_feat_S"]
    n_edge_feat = setup_dict["n_edge_feat_ST"] if is_ST else setup_dict["n_edge_feat_S"]
    other_n_node_feat = setup_dict["n_node_feat_S"] if is_ST else setup_dict["n_node_feat_ST"]

    model_params = valid_block_params(model_dict, setup_dict, layer_dict, n_node_feat, other_n_node_feat, 
                                      idx, is_ST, is_enc, is_dec)   

    n_out_message = setup_dict["n_hid"]
    if setup_dict["model_name"] == "Graph_Attention":
        n_out_message *= setup_dict["n_heads"]
        if "edge_params" in model_params:
            model_params["edge_params"]["n_heads"] = setup_dict["n_heads"]


    for key in model_params.keys():
        if key == "edge_params":           
            if is_enc:
                model_params[key]["n_feat_edge"] = (
                      n_out_message * model_params[key]["with_edges_below"]
                    + n_edge_feat[idx] * model_params[key]["with_edges_own"]
                    + (n_edge_feat[idx+1] if model_params[key]["with_edges_above"] else 0)
                    + n_out_message * model_params[key]["with_edges_depth"])
               
                model_params[key]["n_feat_node"] = (
                      n_node_feat[idx] * model_params[key]["with_nodes_own"]
                    + (n_node_feat[idx+1] if model_params[key]["with_nodes_above"] else 0)
                    + n_out_message * model_params[key]["with_nodes_below"]
                    + n_node_feat[-1] * model_params[key]["with_globals"])
                
                if is_ST: model_params[key]["n_feat_node"] += n_out_message * model_params[key]["with_nodes_depth"]
                else: model_params[key]["n_feat_node"] += other_n_node_feat[idx] * model_params[key]["with_nodes_depth"]
            
            else:
                model_params[key]["n_feat_edge"] = (
                    model_params[key]["with_edges_below"] + model_params[key]["with_edges_own"]
                    + model_params[key]["with_edges_above"] + model_params[key]["with_edges_depth"]) * n_out_message
                
                model_params[key]["n_feat_node"] = model_params[key]["with_globals"] * setup_dict["n_hid"] + (
                      model_params[key]["with_nodes_own"] + model_params[key]["with_nodes_below"]
                    + model_params[key]["with_nodes_above"] + model_params[key]["with_nodes_depth"]) * n_out_message
            
            model_params[key]["n_out"] = n_out_message if model_params[key]["project"] else model_params[key]["n_feat_node"]

        elif key == "node_params":
            edge_n_out = model_params["edge_params"]["n_out"] if "edge_params" in model_params else 0
            if is_enc:
                model_params[key]["n_feat"] = (
                      n_out_message * model_params[key]["with_nodes_below"]
                    + (n_node_feat[idx+1] if model_params[key]["with_nodes_above"] else 0)
                    + n_node_feat[idx] * model_params[key]["with_nodes_own"]
                    + n_node_feat[-1] * model_params[key]["with_globals"]
                    + edge_n_out * model_params[key]["with_receivers"])
                
                if is_ST: model_params[key]["n_feat"] += n_out_message * model_params[key]["with_nodes_depth"]
                else: model_params[key]["n_feat"] += other_n_node_feat[idx] * model_params[key]["with_nodes_depth"]
    
            else:
                model_params[key]["n_feat"] = (
                    model_params[key]["with_nodes_below"] + model_params[key]["with_nodes_own"]
                    + model_params[key]["with_nodes_above"] + model_params[key]["with_nodes_depth"]
                    + model_params[key]["with_globals"]) * n_out_message
                
                model_params[key]["n_feat"] += model_params[key]["with_receivers"] * edge_n_out

            model_params[key]["n_out"] = n_out_message if model_params[key]["project"] else model_params[key]["n_feat"]
        
        elif key == "selection_params":
            model_params["selection_params"]["n_out"] = setup_dict["n_cluster_nodes"][idx]
            model_params[key]["n_feat"] = (
                model_params[key]["with_own_senders"]) * n_out_message

            model_params[key]["n_feat_node"] = model_params["node_params"]["n_out"]
 
    return copy.deepcopy(model_params)

def valid_block_params(model_dict, setup_dict, layer_dict, n_node_feat, other_n_node_feat, 
                        idx, is_ST, is_enc, is_dec):
    
    is_top = (idx == len(n_node_feat)-1)
    has_G_attr = ((n_node_feat[-1] != 0) and len(n_node_feat) > 1) if is_enc else len(n_node_feat) > 1
    is_G = len(n_node_feat) > 1 and setup_dict["n_cluster_nodes"][-1] == 1 and is_top

    is_H = False if idx == 0 else True
    has_attr = (n_node_feat[idx] != 0) if is_enc else True
    has_attr_below = is_H
    has_attr_above = (not is_top and n_node_feat[idx+1] != 0) if is_enc else not is_top
    has_attr_depth = other_n_node_feat[idx] != 0 or is_ST
    
    is_below_top = (idx == len(n_node_feat)-2)
    is_multi = sum(setup_dict["n_node_feat_S"]) > 0 and sum(setup_dict["n_node_feat_ST"]) > 0

    valid_params = {
        "with_nodes_own": has_attr,
        "with_nodes_below": has_attr_below and (not is_dec or setup_dict["return_type"] == "graph_level"),
        "with_nodes_above": has_attr_above and (not is_dec or setup_dict["return_type"] == "node_level"),
        "with_nodes_depth": is_multi and has_attr_depth,
        "with_globals": not is_H and has_G_attr and not is_below_top and (not is_dec or setup_dict["return_type"] == "node_level"),
        "with_receivers": _DEFAULT_NODE_BLOCK_OPT["with_receivers"] and not is_G,
        "with_selection_depth": is_ST,
    }

    params = {}
    if (not is_G):
        params["edge_params"] = merge_dict(_DEFAULT_EDGE_BLOCK_OPT, model_dict["edge_params"])
        params["edge_params"] = make_valid_params(params["edge_params"], layer_dict, False)
        params["edge_params"] = make_valid_params(params["edge_params"], valid_params, True)

    params["node_params"] = merge_dict(_DEFAULT_NODE_BLOCK_OPT, model_dict["node_params"])
    if is_G or setup_dict["model_name"] in ["Graph_Sage_mean", "Graph_Sage_pool"]:
        params["node_params"] = make_valid_params(params["node_params"], model_dict["edge_params"], False)
        params["node_params"] = make_valid_params(params["node_params"], layer_dict, False)
        params["node_params"]["project"] = True
        params["node_params"]["activation"] = model_dict["node_params"]["activation"]
    params["node_params"] = make_valid_params(params["node_params"], valid_params, True)

    if not is_top and not(is_dec and setup_dict["return_type"] == "node_level"):
        params["selection_params"] = merge_dict(_DEFAULT_SELECT_BLOCK_OPT, model_dict["selection_params"])
        params["selection_params"] = make_valid_params(params["selection_params"], layer_dict, False)
        params["selection_params"] = make_valid_params(params["selection_params"], valid_params, True)
        params["selection_params"]["with_own_senders"] = (layer_dict["with_nodes_below"] or layer_dict["with_nodes_above"]) \
                                                            and (not params["selection_params"]["with_selection_depth"] \
                                                            and not params["selection_params"]["with_selection_previous"])
        if (not is_G):
            params["selection_params"]["with_edge_weights"] = params["edge_params"]["with_edge_weights"]
        if (params["selection_params"]["with_selection_depth"] or setup_dict["n_cluster_nodes"][idx] == 1 
            or params["selection_params"]["with_selection_previous"]): 
            params["selection_params"]["project"] = False
        if not (params["selection_params"]["with_own_senders"] or params["selection_params"]["with_selection_depth"]
                or params["selection_params"]["with_selection_previous"]):
            params.pop("selection_params")
        
    return params

def n_features_graphs(dict, n_cluster_nodes):
    dict = dict.copy()

    dict["n_edge_feat_S"] = [0]
    dict["n_node_feat_S"] = [dict["n_node_feat_S"]]
    dict["n_edge_feat_ST"] = [0]
    dict["n_node_feat_ST"] = [dict["n_node_feat_ST"]]

    for _ in range(len(n_cluster_nodes)-1):
        dict["n_edge_feat_S"].append(0)
        dict["n_node_feat_S"].append(0)
        dict["n_edge_feat_ST"].append(0)
        dict["n_node_feat_ST"].append(0)

    if len(n_cluster_nodes) > 0:
        dict["n_edge_feat_S"].append(0)
        dict["n_edge_feat_ST"].append(0)

        if n_cluster_nodes[-1] == 1:
            dict["n_node_feat_S"].append(dict["n_feat_glob_S"])
            dict["n_node_feat_ST"].append(dict["n_feat_glob_ST"])
        else:
            dict["n_node_feat_S"].append(0)
            dict["n_node_feat_ST"].append(0)

    dict["n_cluster_nodes"] = n_cluster_nodes

    return dict

def merge_model_parameters(base_dict, model_dict):
    if "selection_params" not in model_dict: model_dict["selection_params"] = {}
    base_dict = base_dict.copy()
    for key in base_dict.keys():
        base_dict[key] = merge_dict(base_dict[key], model_dict[key])
    return base_dict

def merge_dict(base_dict, add_dict):
    return_dict = base_dict.copy()
    for key, value in add_dict.items():
        return_dict[key] = value
    return return_dict

def make_valid_params(base_params, valid_params, make_valid):
    for key, value in valid_params.items():
        if key in base_params:
            if make_valid:
                base_params[key] = base_params[key] and value
            else:
                base_params[key] = value
    return base_params