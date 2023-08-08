import torch
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import timedelta
from torch_geometric.data import Dataset, Data
import os.path as osp
import json
from sklearn import preprocessing

class FST_Dataset(Dataset):
    
    def __init__(self, root, input_types, adj, seq_len, download=None, transform=None, 
                pre_transform=None, pre_filter=None):
        
        self.root = root
        self.input_types = input_types
        self.adj = adj
        self.seq_len = seq_len

        super(FST_Dataset, self).__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        # self.data, self.slices, self.config_list = torch.load(self.processed_paths[0])
        with open(osp.join(self.processed_dir, "config.txt"), 'r') as file:
            self.config = json.load(file)

    @property
    def raw_file_names(self):
        return ['target_df_aantal_cleaned_new.pkl', 'buurt_polygons.pkl', 'weather.pkl', 
                'bbga_pca.pkl', 'NPR_PRC_Stacked_Occupation.csv']

    @property
    def processed_dir(self) -> str:
        dir = self.input_types + [str(self.seq_len)]
        return osp.join(self.root, 'processed_IST', "_".join(dir))

    @property
    def processed_file_names(self):
        return ["_".join(self.input_types) + "_" + str(s) + ".pt" for s in list(range(len(pd.read_pickle(self.raw_paths[0]))))]

    def process(self):
        target_df = pd.read_pickle(self.raw_paths[0])
        buurt_polygons = pd.read_pickle(self.raw_paths[1])
        T_feature_list, S_feature_list, ST_feature_list = [], [], []
        T_feature_types = set(self.input_types) & {"cyclical_hour", "cyclical_day", "weather"}
        time_range = pd.date_range(start=target_df.index[0] - timedelta(hours=self.seq_len), end=target_df.index[-1], freq='H')
        nfeat_RNN, nfeat_RGCN, nfeat_GCN = 0, 0, 0
        if len(T_feature_types) > 0:
            for input_type in T_feature_types:
                if input_type == "cyclical_hour":
                    feature = np.vstack([(target_df.index - timedelta(hours=x)).hour for x in range(self.seq_len-1, -1, -1)]).T
                    feature = np.stack([np.sin(2*np.pi*feature/24), np.cos(2*np.pi*feature/24)], axis=-1)
                if input_type == "cyclical_day":
                    feature = np.vstack([(target_df.index - timedelta(hours=x)).day for x in range(self.seq_len-1, -1, -1)]).T
                    feature = np.stack([np.sin(2*np.pi*feature/7), np.cos(2*np.pi*feature/7)], axis=-1)
                if input_type == "weather":
                    feature = pd.read_pickle(self.raw_paths[2])
                    feature = feature.reindex(time_range).fillna(method="bfill")
                    feature.iloc[:,:] = preprocessing.scale(feature, axis=1)
                    feature = np.stack([feature.loc[target_df.index - timedelta(hours=x)] for x in range(self.seq_len, 0, -1)], axis = 1)
                T_feature_list.append(torch.Tensor(feature))

            T_feature_list = torch.cat((T_feature_list), -1).unsqueeze(1)
            nfeat_RNN += T_feature_list.shape[-1]

        S_feature_types = set(self.input_types) & {"centroid", "bbga"}
        if len(S_feature_types) > 0:
            for input_type in S_feature_types:
                if input_type == "centroid":
                    feature = torch.Tensor(np.vstack([[p.x, p.y] for p in buurt_polygons['norm_centroid']]))
                    S_feature_list.append(feature.unsqueeze(0).expand(len(target_df), -1, -1))                    
                if input_type == "bbga":
                    feature = pd.read_pickle(self.raw_paths[3])
                    feature = torch.Tensor(np.array(np.split(feature.values.T, len(feature.index.get_level_values(0).unique()), axis=1)))

                    S_feature_list.append(feature[target_df.index.year.values - target_df.index[0].year])
            S_feature_list = torch.cat((S_feature_list), -1)
            nfeat_GCN += S_feature_list.shape[-1]

        ST_feature_types = set(self.input_types) & {"npr", "historical"}
        if len(ST_feature_types) > 0:
            for input_type in ST_feature_types:
                if input_type == "npr":
                    feature = pd.read_csv(self.raw_paths[4], sep = ';')
                    feature["buurtcode"] = feature["buurtcode"].str.lower()
                    feature = pd.melt(feature, id_vars=['buurtcode', 'B_TYD_V_RECHT'], value_vars=feature.columns[3:], value_name='count')
                    feature['interval_start'] = pd.to_datetime(feature['B_TYD_V_RECHT'] + '-' + feature['variable'].str[5:], format='%Y-%m-%d-%H')
                    feature = feature.groupby(['buurtcode', 'interval_start']).sum().unstack(level=0)
                    feature = feature.droplevel(level=0, axis=1)
                    feature = feature.reindex(time_range, columns=target_df.columns)
                    feature.iloc[:,:] = preprocessing.scale(feature, axis=1)
                if input_type == "historical":
                    feature = target_df.reindex(time_range)

                feature = [feature.loc[target_df.index - timedelta(hours=x)].T for x in range(self.seq_len, 0, -1)]
                ST_feature_list.append(torch.cat((torch.Tensor(np.stack([df.fillna(0).T for df in feature], axis= 2)).unsqueeze(-1), 
                                        torch.Tensor(np.stack([df.notnull().T for df in feature], axis= 2)).unsqueeze(-1)), axis=-1))
            
            ST_feature_list = torch.cat((ST_feature_list), -1)
            nfeat_RGCN += ST_feature_list.shape[-1]

        self.config = {"n_node_feat_S": nfeat_GCN, "n_node_feat_ST": nfeat_RGCN, "n_feat_glob_ST": nfeat_RNN,
                       "n_edge_feat_S": 0, "n_edge_feat_ST": 0, "n_feat_glob_S": 0, "n_out_final": 1}
        
        y = torch.Tensor(target_df.values)

        mask = torch.BoolTensor(target_df.notnull().reindex(time_range).loc[target_df.index].values.astype(bool))

        for i in range(len(target_df)):

            data = Data(edge_index = self.adj, num_nodes = y.shape[-1],
                        y = y[i][mask[i]].clone(), mask = mask[i].clone())
            if len(T_feature_list) != 0: data.g_t = T_feature_list[i].clone() 
            if len(S_feature_list) != 0: data.v_s = S_feature_list[i].clone()
            if len(ST_feature_list) != 0: data.v_st = ST_feature_list[i].clone()
            torch.save(data, osp.join(self.processed_dir, "_".join(self.input_types) + "_" + str(i) + ".pt"))
        with open(osp.join(self.processed_dir, "config.txt"), 'w') as file:
            json.dump(self.config, file)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, "_".join(self.input_types) + "_" + str(idx) + ".pt"))
        return data
