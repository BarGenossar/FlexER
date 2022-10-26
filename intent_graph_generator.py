import faiss
from torch_geometric.data import Dataset, Data, HeteroData
import numpy as np
from utils import *
import torch
import re


class IntentGraphReader(Dataset):
    def __init__(self, task, intents_list, generate_neighbors_dict, k_size,
                 ind_method, file_type, file_types, seed, method="IndexFlatL2"):
        super(IntentGraphReader, self).__init__()
        self.task = task
        self.intents_list = intents_list
        self.generate_neighbors_dict = generate_neighbors_dict
        self.k_size = k_size
        self.ind_method = ind_method
        self.file_type = file_type
        self.file_types = file_types
        self.method = method
        self.seed = seed
        self.main_data_path, self.seed_data_path = self.create_data_paths()
        self.file_sizes = self.get_file_sizes()
        self.node_features_dict = self.get_node_features()
        self.labels_dict = self.get_labels()
        self.edges_dict = self.get_edges()
        self.generate_graph()

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return

    def generate_graph(self):
        data = HeteroData()
        data = self.define_graph_nodes_and_labels(data)
        data = self.define_graph_edges(data)
        torch.save(data, ''.join((self.seed_data_path, '_', self.file_type, '_data.pt')))
        return

    def define_graph_nodes_and_labels(self, data):
        for intent in self.intents_list:
            data[''.join(('intent', str(intent)))].x = self.node_features_dict[intent]
            data[''.join(('intent', str(intent)))].y = self.labels_dict[intent]
        return data

    def define_graph_edges(self, data):
        for intent1 in self.intents_list:
            intent1_str = ''.join(('intent', str(intent1)))
            for intent2 in self.intents_list:
                if intent2 < intent1 or (self.k_size == 0 and intent1 == intent2):
                    continue
                intent2_str = ''.join(('intent', str(intent2)))
                data[intent1_str, ''.join((intent1_str, 'TO', intent2_str)), intent2_str].edge_index = \
                    self.edges_dict[(intent1, intent2)]
                data[intent2_str, ''.join((intent2_str, 'TO', intent1_str)), intent1_str].edge_index = \
                    self.edges_dict[(intent2, intent1)]
        return data

    def open_file(self, file_type, intent):
        current_path = ''.join((self.main_data_path, "_", file_type, str(intent), ".txt"))
        current_file = open(current_path, "r", encoding="utf-8")
        file_lines = current_file.readlines()
        current_file.close()
        return file_lines

    def get_labels(self):
        labels_dict = dict()
        for intent in self.intents_list:
            labels = []
            files_lines = self.open_file(self.file_type, intent)
            for id_val, line in enumerate(files_lines):
                try:
                    labels.append(int(re.sub("[^0-9]", "", line[-2])))
                except:
                    labels.append(int(re.sub("[^0-9]", "", line[-1])))
            labels_dict[intent] = torch.tensor(labels, dtype=torch.float)
        return labels_dict

    def get_node_features(self):
        poolers_paths_dict = self.create_poolers_path_dict()
        poolers_dict = dict()
        for intent, poolers_path in poolers_paths_dict.items():
            poolers_file = open(poolers_path, "r", encoding="utf-8")
            lines_preds = poolers_file.readlines()
            node_features = []
            for id_val, line in enumerate(lines_preds):
                pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
                node_features.append(np.array(list(map(float, pooler.split(' ')))))
            poolers_dict[intent] = torch.tensor(np.array(node_features), dtype=torch.float)
        return poolers_dict

    def get_edges(self):
        total_nodes = self.file_sizes[self.file_type]
        neighbors_dict = self.get_neighbors_dict()
        edges_dict = self.create_intra_edges(neighbors_dict)
        edges_dict = self.create_inter_edges(edges_dict, total_nodes)
        return edges_dict

    def create_data_paths(self):
        if 'WDC' in self.task or 'Amazon-Website' in self.task:
            main_data_path = ''.join(("data/", self.task, "/", self.task.split('/')[1]))
            seed_data_path = ''.join(("data/", self.task, "/", str(self.seed), "/", self.task.split('/')[1]))
        elif "Structured" in self.task:
            main_data_path = ''.join(("data/er_magellan/", self.task, "/", self.task.split('/')[1]))
            seed_data_path = ''.join(("data/er_magellan/", self.task, "/", str(self.seed), "/", self.task.split('/')[1]))
        else:
            raise ValueError("Data path is not valid!")
        return main_data_path, seed_data_path

    def get_file_sizes(self):
        sizes_dict = dict()
        for file_type in self.file_types:
            file_lines = self.open_file(file_type, 0)
            sizes_dict[file_type] = len(file_lines)
        sizes_dict['total'] = sum(sizes_dict.values())
        return sizes_dict

    def create_poolers_path_dict(self):
        poolers_paths_dict = dict()
        for intent in self.intents_list:
            poolers_paths_dict[intent] = ''.join((self.seed_data_path,
                                                  "_", self.file_type,
                                                  str(intent),
                                                  "_output.txt"))
        return poolers_paths_dict

    def get_neighbors_dict(self):
        if self.generate_neighbors_dict:
            neighbors_dict = self.create_neighbors_dict()
        else:
            neighbors_dict = load_pkl(''.join((self.seed_data_path, '_', self.file_type, '_neighbors_dict.pkl')))
        return neighbors_dict

    def create_neighbors_dict(self):
        neighbors_dict = dict()
        d = len(self.node_features_dict[0][0])
        for intent, poolers in self.node_features_dict.items():
            curr_neighbors_dict = dict()
            index = self.find_index_method(d)
            poolers = np.array(poolers, dtype="float32")
            index.add(poolers)
            for pooler_id, pooler_vec in enumerate(poolers):
                query_pooler = np.expand_dims(pooler_vec, axis=0)
                dists, neighbors = index.search(query_pooler, 50)
                curr_neighbors_dict[pooler_id] = [neighbor for neighbor in neighbors[0][1:]]
            neighbors_dict[intent] = curr_neighbors_dict
        save_to_pkl(neighbors_dict, ''.join((self.seed_data_path, '_', self.file_type, '_neighbors_dict.pkl')))
        return neighbors_dict

    def find_index_method(self, d):
        if self.method == "flat_IP":
            return faiss.IndexFlatIP(d)
        elif self.method == "IndexFlatL2":
            return faiss.IndexFlatL2(d)
        else:
            # TODO: implement
            return

    @staticmethod
    def calc_weights(dists):
        dists_sum = dists.sum()
        inv_norm_dists = [(dists_sum - dist) / dists_sum for dist in dists[0][1:]]
        inv_norm_dists_sum = sum(inv_norm_dists)
        return [val / inv_norm_dists_sum for val in inv_norm_dists]

    def create_intra_edges(self, neighbors_dict):
        edges_dict = dict()
        if self.k_size > 0:
            for intent, neighbors_dict_intent in neighbors_dict.items():
                edges_list = []
                for pooler_id, neighbors in neighbors_dict_intent.items():
                    edges_list.extend([[neighbor, pooler_id]
                                       for neighbor in neighbors[:self.k_size]])
                edges_dict[(intent, intent)] = torch.tensor(edges_list, dtype=torch.long).t()
        return edges_dict

    def create_inter_edges(self, edges_dict, total_nodes):
        inter_edges_list = [[pooler_id, pooler_id] for pooler_id in range(total_nodes)]
        for intent1 in self.intents_list:
            for intent2 in self.intents_list:
                if intent2 <= intent1:
                    continue
                edges_dict[(intent1, intent2)] = torch.tensor(inter_edges_list, dtype=torch.long).t()
                edges_dict[(intent2, intent1)] = torch.tensor(inter_edges_list, dtype=torch.long).t()
        return edges_dict

