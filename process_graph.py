import torch
from torch_geometric.data import Data
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import pandas as pd
from itertools import combinations
import torch_geometric
torch_geometric.set_debug(True)


class GraphReaderFlexER:
    def __init__(self, file_type, task, files_path, intent, intents_num):
        self.file_type = file_type
        self.task = task
        self.graph_type = graph_type
        self.clean_task = task.split('/')[1]
        self.files_path = files_path
        self.intents_num = intents_num
        self.emb_type = emb_type
        self.inference = inference
        self.MCML_inference = MCML_inference
        self.df = self.create_df()
        self.poolers = self.create_poolers()
        self.graph_idxs = self.df.index.values
        self.node_embeddings = self.create_node_embeddings()
        self.edges, self.raw_pairs = self.create_edges()
        self.labels = self.create_labels(intent)
        self.correlations = correlations
        self.edge_weights, correlations = self.create_edges_weights()
        self.dataset = self.create_dataset()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def create_df(self):
        results_dict = self.read_predictions()
        results_dict = self.read_labels(results_dict)
        df = pd.DataFrame.from_dict(results_dict)
        df = self.convert_confidence_to_pred(df)
        df.reset_index(inplace=True)
        df.to_csv("output/" + self.task + "/" +
                  self.clean_task + '_' + self.file_type + "_"
                  + self.emb_type + ".csv")
        return df

    def read_predictions(self):
        results_dict = dict()
        for intent in range(self.intents_num):
            if self.inference == 'MCML' and self.MCML_inference != 'Multilabel':
                if 'Structured' in self.task:
                    preds_file = open(self.files_path + 'er_magellan/' +
                                      self.task + '/' + self.clean_task +
                                      '_' + self.file_type + '_' + str(intent) +
                                      '_MCML_output.txt', "r", encoding="utf-8")
                else:
                    preds_file = open(self.files_path + self.task + '/'
                                      + self.clean_task + '_'
                                      + self.file_type + '_' + str(intent) +
                                      '_MCML_output.txt', "r", encoding="utf-8")

            else:
                if 'Structured' in self.task:
                    preds_file = open(self.files_path + 'er_magellan/' +
                         self.task + '/' + self.clean_task + '_'
                         + self.file_type + str(intent) +
                         '_output.txt', "r", encoding="utf-8")
                else:
                    preds_file = open(self.files_path + self.task + '/'
                                      + self.clean_task + '_'
                                      + self.file_type + str(intent) +
                                      '_output.txt', "r", encoding="utf-8")
            Lines_preds = preds_file.readlines()
            results_dict[intent] = {}
            for id_val, line in enumerate(Lines_preds):
                confidence = float(re.sub("[^0-9.]",
                                          "", line.split('match_confidence')[1].split(',')[0]))
                prediction = int(line.split('"prediction":')[1][2])
                # pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
                if prediction:
                    results_dict[intent][id_val] = confidence
                else:
                    results_dict[intent][id_val] = 1 - confidence
            preds_file.close()
        return results_dict

    def create_poolers(self):
        if self.emb_type != 'poolers':
            return None
        poolers_dict = dict()
        for intent in range(self.intents_num):
            if self.inference == 'MCML' and self.MCML_inference != 'Multilabel':
                if 'Structured' in self.task:
                    preds_file = open(self.files_path + 'er_magellan/' +
                                      self.task + '/' + self.clean_task +
                                      '_' + self.file_type + '_' + str(intent) +
                                      '_MCML_output.txt', "r", encoding="utf-8")
                else:
                    preds_file = open(self.files_path + self.task + '/'
                                      + self.clean_task + '_'
                                      + self.file_type + '_' + str(intent) +
                                      '_MCML_output.txt', "r", encoding="utf-8")

            else:
                if 'Structured' in self.task:
                    preds_file = open(self.files_path + 'er_magellan/' + self.task + '/'
                                      + self.clean_task + '_'
                                      + self.file_type + str(intent) +
                                      '_output.txt', "r", encoding="utf-8")
                else:
                    preds_file = open(self.files_path + self.task + '/'
                                      + self.clean_task + '_'
                                      + self.file_type + str(intent) +
                                      '_output.txt', "r", encoding="utf-8")
            Lines_preds = preds_file.readlines()
            poolers_dict[intent] = {}
            for id_val, line in enumerate(Lines_preds):
                pooler = line.split('pooler')[1][3:-2].replace('[', '').replace(',', '').replace(']', '')
                poolers_dict[intent][id_val] = list(map(float, pooler.split(' ')))
            preds_file.close()
        return poolers_dict

    def read_labels(self, results_dict):
        for intent in range(self.intents_num):
            if 'Structured' in self.task:
                ground_truth_file = open(self.files_path + 'er_magellan/' +
                                         self.task + '/' + self.clean_task + '_'
                                         + self.file_type + str(intent) +
                                         '.txt', "r", encoding="utf-8")
            else:
                ground_truth_file = open(self.files_path + self.task + '/'
                                         + self.clean_task + '_'
                                         + self.file_type + str(intent) +
                                         '.txt', "r", encoding="utf-8")
            Lines_ground_truth = ground_truth_file.readlines()
            results_dict['label' + str(intent)] = {}
            for id_val, line in enumerate(Lines_ground_truth):
                try:
                    results_dict['label' + str(intent)][id_val] = int(line[-2])
                except:
                    results_dict['label' + str(intent)][id_val] = int(line[-1])
            ground_truth_file.close()
        return results_dict

    def convert_confidence_to_pred(self, df):
        for intent in range(self.intents_num):
            df['prediction' + str(intent)] = df[intent].apply(lambda x: 1 if x > 0.5 else 0)
        return df

    def create_node_embeddings(self):
        if self.emb_type == 'probs':
            # Use basic model probabilities as node embeddings
            return self.probs_embeddings_dict()
        else:
            # Use the pooler tensor as node embeddings
            return self.poolers_embeddings_dict()

    def probs_embeddings_dict(self):
        node_embeddings_dict = dict()
        for idx in self.graph_idxs:
            graph_embeddings = []
            for intent in range(self.intents_num):
                match_pred = self.df.iloc[idx][intent]
                graph_embeddings.append([1 - match_pred, match_pred])
            node_embeddings_dict[idx] = torch.tensor(graph_embeddings, dtype=torch.float)
        return node_embeddings_dict

    def poolers_embeddings_dict(self):
        node_embeddings_dict = dict()
        for idx in self.graph_idxs:
            graph_embeddings = []
            for intent in range(self.intents_num):
                graph_embeddings.append(self.poolers[intent][idx])
            node_embeddings_dict[idx] = torch.tensor(graph_embeddings, dtype=torch.float)
        return node_embeddings_dict

    def create_edges(self):
        edge_index_dict = dict()
        if self.graph_type == 'complete':
            pairs = list(list(pair) for pair in combinations([intent for intent in
                                                              range(self.intents_num)], 2))

            pairs.extend(list(list(pair) for pair in combinations([intent for intent in
                                                                   range(self.intents_num - 1,
                                                                         -1,  -1)], 2)))

        else:
            # Tree- The user must handle it manually
            if 'Amazon-Website' in self.task:
                pairs = [[0, 1], [1, 0], [0, 2], [2, 0], [2, 3], [3, 2], [2, 4], [4, 2]]
            elif 'iTunes-Website' in self.task:
                pairs = [[0, 1], [1, 0], [0, 2], [2, 0], [2, 3], [3, 2]]
            elif 'Walmart-Amazon' in self.task:
                pairs = [[0, 1], [1, 0], [1, 1], [2, 1], [1, 3], [3, 1]]
            else:
                # WDC
                pairs = [[0, 1], [1, 0]]
        for idx in self.graph_idxs:
            edge_index = []
            for pair in pairs:
                edge_index.append(pair)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_index_dict[idx] = edge_index
        return edge_index_dict, pairs

    def create_labels(self, intent):
        labels_dict = dict()
        for idx in self.graph_idxs:
            label = torch.tensor(int(self.df.iloc[idx]['label'+str(intent)]))
            labels_dict[idx] = int(label)
        return labels_dict

    def create_edges_weights(self):
        edge_weights_list = []
        if self.correlations is None:
            # This is the training set
            correlations = dict()
            col_pairs = list(combinations([intent for intent in
                                           range(self.intents_num)], 2))

            col_pairs.extend(list(combinations([intent for intent in
                                                range(self.intents_num - 1,
                                                      -1,  -1)], 2)))

            for pair in col_pairs:
                correlations[pair] = round(self.df['label' + str(pair[0])].
                                           corr(self.df['label' + str(pair[1])]), 4)
        else:
            correlations = self.correlations
        for idx in self.graph_idxs:
            weights = [[correlations[tuple(pair)]] for pair in self.raw_pairs]
            edge_weights_list.append(torch.tensor(weights, dtype=torch.float))
        return edge_weights_list, correlations

    @property
    def get_correlations(self):
        return self.correlations

    def create_dataset(self):
        data_objects_list = []
        torch.manual_seed(1)
        for idx in self.graph_idxs:
            x = self.node_embeddings[idx]
            edges = self.edges[idx]
            edge_weight = self.edge_weights[idx]
            y = self.labels[idx]
            data_object = Data(x=x, edge_index=edges, edge_attr=edge_weight, y=y)
            data_objects_list.append(data_object)
        return data_objects_list
