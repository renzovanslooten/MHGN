import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Dropout
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from layers import *
from collections import OrderedDict
import pandas as pd
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import A3TGCN2

#-----------------------------------------------------------------------------------------------------------------------------------------#

class RMGN(torch.nn.Module):
    def __init__(self, model_name, setup_dict, data_dict, n_cluster_nodes, hierarchical_network_params):
        super().__init__()
        self.model_name = model_name
        self.setup_dict = setup_dict
        self.data_dict = data_dict
        self.n_cluster_nodes = n_cluster_nodes
        self.params = hierarchical_network_params
        self.build()

    def build(self):     
        self.S_encoder, self.S_processor, self.S_decoder = [], [], []
        self.ST_encoder, self.ST_processor, self.ST_decoder = [], [], []
        
        n_hid = self.setup_dict["n_hid"]
        if self.model_name == "Graph_Attention": n_hid *= self.setup_dict["n_heads"]
        
        for key, value in self.params.items():
            if key == "params_S":
                if "enc_params" in value:
                    self.S_encoder = nn.ModuleList([GraphNetwork(params, key) for params in value["enc_params"]])
                else:
                    self.S_encoder = nn.Sequential(Linear(self.data_dict["n_node_feat_S"], n_hid, bias=self.setup_dict["emb_bias"]),
                                                   nn.ReLU())
                
                if "pro_params" in value:
                    for _ in range(self.setup_dict["n_hid_layer"]):
                        self.S_processor.append(nn.ModuleList([GraphNetwork(params, key) for params in value["pro_params"]]))
                    self.S_processor = nn.ModuleList(self.S_processor)
        
                if "dec_params" in value:
                    self.S_decoder = nn.ModuleList([GraphNetwork(params, key) for params in value["dec_params"]])
            if key == "params_ST":
                if "enc_params" in value:
                    self.ST_encoder = nn.ModuleList([GraphNetwork(params, key) for params in value["enc_params"]])
                
                if "pro_params" in value:
                    for _ in range(self.setup_dict["n_hid_layer"]):
                        self.ST_processor.append(nn.ModuleList([GraphNetwork(params, key) for params in value["pro_params"]]))
                    self.ST_processor = nn.ModuleList(self.ST_processor)

        self.after_model = nn.Sequential(Linear(n_hid, n_hid//2),
                                         nn.ReLU(), 
                                         Linear(n_hid//2, n_hid//4), 
                                         nn.ReLU(), 
                                         Linear(n_hid//4, self.data_dict["n_out_final"]))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.S_encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.after_model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, S_graphs, ST_graphs):
        if self.setup_dict["HMGN_emb"]:
            for i in range(len(self.S_encoder)):
                S_graphs = self.S_encoder[i](S_graphs, ST_graphs, i)  
            for i in range(len(self.ST_encoder)):
                ST_graphs = self.ST_encoder[i](ST_graphs, S_graphs, i)
        else:
            S_graphs[0].node_attr = self.S_encoder(S_graphs[0].node_attr)

        for j in range(self.setup_dict["n_hid_layer"]):
            for i in range(len(self.S_processor[j])):
                S_graphs = self.S_processor[j][i](S_graphs, ST_graphs, i)
            if self.ST_processor:
                for i in range(len(self.ST_processor[j])):
                    ST_graphs = self.ST_processor[j][i](ST_graphs, S_graphs, i)    
        
        if self.setup_dict["return_type"] == "node_level": 
            range_s_decoder = range(len(self.S_decoder)-1, -1, -1)
            range_st_decoder = range(len(self.ST_decoder)-1, -1, -1)
        else:
            range_s_decoder = range(len(self.S_decoder))
            range_st_decoder = range(len(self.ST_decoder))       

        for i in range_s_decoder:
            S_graphs = self.S_decoder[i](S_graphs, ST_graphs, i)
        for i in range_st_decoder:
            S_graphs = self.ST_decoder[i](ST_graphs, S_graphs, i)
    
        loss_1 = sum([graph.loss_1 for graph in S_graphs])
        loss_2 = sum([graph.loss_2 for graph in S_graphs]) 

        if self.setup_dict["return_type"] == "node_level":
            return (self.after_model(S_graphs[0].node_attr), loss_1, loss_2)
        else:
            if self.n_cluster_nodes and self.n_cluster_nodes[-1] == 1:
                return (self.after_model(S_graphs[-1].node_attr), loss_1, loss_2)
            else:
                return (self.after_model(global_mean_pool(S_graphs[-1].node_attr, S_graphs[-1].batch)), loss_1, loss_2)
                # return (self.after_model(global_mean_pool(S_graphs[-1].node_attr, S_graphs[-1].batch)), S_graphs[-1].mincut_loss, S_graphs[-1].ortho_loss)

#-----------------------------------------------------------------------------------------------------------------------------------------#

class GraphNetwork(torch.nn.Module):
    def __init__(self, block_params, data_type):
        super(GraphNetwork, self).__init__()
        self.block_params = block_params

        def get_project_fn(data_type, element_params, n_feat, n_out):
            if n_feat == 0 or not element_params["project"]:
                return None
            if "selection_meth" in element_params:
                if element_params["selection_meth"] == "gcn":
                    new_element_params = {k: v for k, v in self.block_params.items() if k != "selection_params"}
                    new_element_params["edge_params"]["n_feat_node"] = new_element_params["edge_params"]["n_out"]
                    new_element_params["node_params"]["normalize"] = False
                    new_element_params["node_params"]["activation"] = None
                    new_element_params["node_params"]["n_feat"] = 128
                    return GraphNetwork(new_element_params, data_type)

            if data_type == "params_S":
                return nn.Linear(n_feat, n_out, bias = element_params["bias"])
            elif data_type == "params_ST":
                return nn.GRU(n_feat, n_out, bias = element_params["bias"])
            else:
                raise NotImplementedError("Not a valid data type")
            
        def get_activation_fn(element_params):
            activation = element_params["activation"] if "activation" in element_params else None
            if activation is not None:
                if activation == "relu":
                    return nn.ReLU()
                elif activation == "sigmoid":
                    return nn.Sigmoid()
                else:
                    raise NotImplementedError("This activation function is not implemented")

        self.model = nn.ModuleList()
        if "edge_params" in block_params.keys():
            n_out = block_params["edge_params"]["n_out"]

            if block_params["edge_params"]["n_feat_edge"] == 0:
                edge_project_fn = None
            else:
                edge_project_fn = get_project_fn(data_type, block_params["edge_params"],
                                                 block_params["edge_params"]["n_feat_edge"], n_out)
            
            node_project_fn = get_project_fn(data_type, block_params["edge_params"],
                                             block_params["edge_params"]["n_feat_node"], n_out)
            
            activation_fn = get_activation_fn(block_params["edge_params"])
            
            self.model.append(EdgeModel(edge_project_fn, node_project_fn, activation_fn, **block_params["edge_params"]))

        if "node_params" in block_params.keys():
            n_out = block_params["node_params"]["n_out"]
            
            node_project_fn = get_project_fn(data_type, block_params["node_params"],
                                             block_params["node_params"]["n_feat"], n_out)
            
            activation_fn = get_activation_fn(block_params["node_params"])
            
            self.model.append(NodeModel(node_project_fn, activation_fn, **block_params["node_params"]))

        if "selection_params" in block_params.keys():
            n_out = block_params["selection_params"]["n_out"]
            if (data_type == "params_S" and block_params["selection_params"]["n_out"] != 0):
                project_fn = get_project_fn(data_type, block_params["selection_params"],
                                                block_params["selection_params"]["n_feat"], n_out)
            else:
                project_fn = None
            
            self.model.append(SelectionModel(project_fn, **block_params["selection_params"]))
            
    def forward(self, graphs, graphs_oth, idx, concat_graphs=None, concat_graphs_oth=None):
        for layer in self.model:
            graphs = layer(graphs, graphs_oth, idx, concat_graphs, concat_graphs_oth)
        return graphs

#-----------------------------------------------------------------------------------------------------------------------------------------#

class GCNRITSI(nn.Module):
    def __init__(self, data_config, model_config, after_config):
        super(GCNRITSI, self).__init__()
        self.config_dict = {**data_config, **model_config, **after_config}
       
        self.build()

    def build(self):
        if self.config_dict["pred_meth"] == "MLP": 
            self.gamma_x = TemporalDecay(1, 1)
        if self.config_dict["h_decay"]: 
            self.h_decay = TemporalDecay(1, self.config_dict["nhid_LSTM"])
        
        nfeat_aftermodel = 0
        nfeat_lstmmodel = self.config_dict["nfeat_LSTM"]
        
        if self.config_dict["S_data"]:
            nfeat_aftermodel += self.config_dict["nout_GCN"]
            self.GCN_Module = GCN3(self.config_dict["seq_len"], self.config_dict["n_nodes"], 
            True, **self.config_dict["GCN_model"])

        if self.config_dict["ST_data"] or self.config_dict["historical_data"]:

            nfeat_lstmmodel += self.config_dict["nout_RGCN"]
            self.RGCN_Module = GCN3(self.config_dict["seq_len"],
             self.config_dict["n_nodes"], False, **self.config_dict["RGCN_model"])

        if self.config_dict["T_data"] or self.config_dict["ST_data"] or self.config_dict["historical_data"]:
            nfeat_aftermodel += self.config_dict["nhid_LSTM"]

            if self.config_dict["RNN_type"] == "GRU":
                self.RNN_CELL = nn.GRUCell(nfeat_lstmmodel, self.config_dict["nhid_LSTM"])
            elif self.config_dict["RNN_type"] == "LSTM":
                self.RNN_CELL = nn.LSTMCell(nfeat_lstmmodel, self.config_dict["nhid_LSTM"])

        self.After_Module = Sequential (nn.Linear(nfeat_aftermodel, self.config_dict["nhid"]),
                                    nn.Dropout(p=self.config_dict["dropout"], inplace=False),
                                    nn.ReLU(),
                                    nn.Linear(self.config_dict['nhid'], self.config_dict['nout'])
                                    )
        
    def forward(self, batch, adj):
        batch_size = len(batch.batch)
        h = Variable(torch.zeros((batch_size, self.config_dict["nhid_LSTM"])))
        c = Variable(torch.zeros((batch_size, self.config_dict["nhid_LSTM"])))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        if "S_data" in batch.keys:
            output_GCN = self.GCN_Module(batch)
            output_GCN = output_GCN.contiguous().view(-1, output_GCN.shape[-1])

        imputations = []
        for t in range(self.config_dict["seq_len"]):
            mask = batch.mask[:, t]
                      
            if self.config_dict["impute"]:               
                if self.config_dict["h_decay"]:
                    delta = batch.delta[:, t, :].reshape(-1, 1)
                    gamma = self.h_decay(delta)
                    h = h * gamma

                if self.config_dict["pred_meth"] == "MLP":
                    input_After = h
                    if "S_data" in batch.keys:
                        input_After = torch.cat(([input_After, output_GCN]), -1)       
                    imputations = self.After_Module(input_After)
                    imputations = imputations.view(mask.shape, -1)

                if self.config_dict["pred_meth"] == "lin_reg":
                    delta = batch.delta[:, t, :].reshape(-1, 1)
                    gamma_x = self.gamma_x(delta)
                    gamma_x = gamma_x.view(mask.shape, -1)
                    imputations = (gamma_x * batch.last_y[:, t, :] + (1 - gamma_x) * batch.mean_y[:, t, :])

                batch.historical[:, t, :, 0] =  batch.historical[:, t, :, 1].bool() * batch.historical[:, t, :, 0] + \
                                                    ~batch.historical[:, t, :, 1].bool() * imputations

            input_RGCN = torch.zeros((0)).to(h.device)
            if "ST_data" in batch.keys:
                input_RGCN = torch.cat(([input_RGCN, batch.ST_data[:, t, :]]), -1)
            if self.config_dict["historical_data"]:
                input_RGCN = torch.cat(([input_RGCN, batch.historical[:, t, :]]), -1)
            
            if len(input_RGCN) > 0:
                output_RGCN = self.RGCN_Module(batch)

            input_RNN = torch.zeros((0)).to(h.device)
            if self.config_dict["T_data"]:
                input_RNN = torch.cat(([input_RNN, batch.T_data[:, t, :].repeat_interleave(batch.ptr[1], dim=0)]), -1)
            if self.config_dict["ST_data"] or self.config_dict["historical_data"]:
                input_RNN = torch.cat(([input_RNN, output_RGCN]), -1)
                
            if len(input_RNN) > 0:
                # h, c = self.RNN_CELL(ST_data.view(-1, ST_data.shape[-1]), (h, c))
                h = self.RNN_CELL(input_RNN.view(-1, input_RNN.shape[-1]), h)  
        
        if "S_data" in batch.keys:
            if len(input_RNN) > 0:
                input_After = torch.cat(([output_GCN, h]), -1)
            else: input_After = output_GCN
        else:
            input_After = h

        return self.After_Module(input_After[mask.flatten()])

#-----------------------------------------------------------------------------------------------------------------------------------------#

class Different_Models(nn.Module):
    def __init__(self, data_config, model_config, after_config):
        super(Different_Models, self).__init__()
        self.config_dict = {**data_config, **model_config, **after_config}
       
        self.build()

    def build(self):
        self.model = A3TGCN2(102,256,8,32)
        self.aftermodel = Linear(256, 1)
        self.activation = nn.ReLU()
        self.S_encoder = TGCN2(102, 64, 32)


    def forward(self, S_graphs, ST_graphs):
        node_attr = S_graphs[0].node_attr.unsqueeze(1).repeat(1, self.dict["seq_len"], 1)
        node_attr = torch.cat((node_attr, ST_graphs[0].node_attr), dim=-1).view(32, 481, 8, 87)
        node_attr = torch.cat((node_attr, ST_graphs[-1].node_attr.unsqueeze(1).repeat(1,481,1,1)), dim=-1).permute(0,1,3,2)
        edge_index = S_graphs[0].edge_index[:, :3012]

        out = self.model(node_attr, edge_index)
        out = self.activation(out)
        out = self.aftermodel(out.flatten(end_dim=1))
        return out
    
    def forward(self, graphs, graphs_oth, idx, concat_graphs=None, concat_graphs_oth=None):
        for layer in self.model:
            graphs = layer(graphs, graphs_oth, idx, concat_graphs, concat_graphs_oth)
        return graphs

#-----------------------------------------------------------------------------------------------------------------------------------------#

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.build(nfeat, nhid, nout, dropout)

    def build(self, nfeat, nhid, nout, dropout):
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

#-----------------------------------------------------------------------------------------------------------------------------------------#

class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.build(nfeat, nhid, nout, dropout)

    def build(self, nfeat, nhid, nout, dropout):
        self.gc1 = MetaLayer(
                        EdgeModel(n_features=nfeat, n_edge_features=0, n_global_features=0, n_hid=nhid, n_out=nout), 
                        NodeModel(n_features=nfeat, n_edge_features=0, n_global_features=0, n_hid=nhid, n_out=nout), 
                        GlobalModel(n_features=nfeat, n_global_features=0, n_hid=nhid, n_out=nout))

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x

#-----------------------------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------------------------#

class SCGCN(nn.Module):
    def __init__(self, nfeat, n_nodes, nout, dropout, n_cluster, cluster_meth):
        super(SCGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.cluster_conv = GraphConvolution(nfeat, nout)
        
        self.cluster_meth = cluster_meth
        if cluster_meth == "linear":
            self.soft_matrix = nn.Linear(nfeat, n_cluster, bias=True)
        elif cluster_meth == "attention":
            self.soft_matrix = nn.Parameter(torch.empty(n_nodes, n_cluster))
            nn.init.xavier_uniform_(self.soft_matrix.data, gain=1.414)
        
   
    def apply_bn(self, x):
        # Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        x = bn_module(x)
        return x

    def forward(self, x, adj):
        """
        :param X_lots: Concat of the outputs of CxtConv and PA_approximation (batch_size, N, in_features).
        :param adj: adj_merge (N, N).
        :return: Output soft clustering representation for each parking lot of shape (batch_size, N, out_features).
        """
        x = self.dropout(x) # (B, N, F)
        if self.cluster_meth == "linear":
            S = F.softmax(self.soft_matrix(x), dim=-1) #(B, N, C)
        elif self.cluster_meth == "attention":
            S = F.softmax(self.soft_matrix, dim=-1).repeat(x.shape[0],1,1)
        X_s = torch.bmm(S.T,x) # (B, K, F)
        
        # GCN
        X_s_c = self.cluster_conv(X_s,adj_score)
        X_s_c = self.dropout(X_s_c) # (B, K, F)
        X_s_c = torch.bmm(S,X_s_c)
        X_s_c = self.apply_bn(X_s_c)
        return X_s_c

#-----------------------------------------------------------------------------------------------------------------------------------------#

class TemporalDecay(nn.Module):
    def __init__(self, input_size, rnn_hid_size):
        super(TemporalDecay, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(self.rnn_hid_size, input_size))
        self.b = Parameter(torch.Tensor(self.rnn_hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-torch.max(0,gamma))
        return gamma
    

