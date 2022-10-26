from intent_graph_generator import IntentGraphReader
from flexer_train_and_evaluate import train_and_evaluate, create_data_path_out
import argparse
import torch
from itertools import combinations


def create_dataloaders_inference(task, intent_list, generate_neighbors_dict, k_size,
                                 file_types, ind_method, seed, load_datasets):
    if generate_neighbors_dict or load_datasets:
        IntentGraphReader(task, intent_list, generate_neighbors_dict, k_size,
                          ind_method, 'train', file_types, seed, ind_method)
        IntentGraphReader(task, intent_list, generate_neighbors_dict, k_size,
                          ind_method, 'valid', file_types, seed, ind_method)
        IntentGraphReader(task, intent_list, generate_neighbors_dict, k_size,
                          ind_method, 'test', file_types, seed, ind_method)
    data_path = create_data_path_out(task, seed)
    train_datareader = torch.load(''.join((data_path, '_train_data.pt')))
    valid_datareader = torch.load(''.join((data_path, '_valid_data.pt')))
    test_datareader = torch.load(''.join((data_path, '_test_data.pt')))
    return train_datareader, valid_datareader, test_datareader


def get_intent_lists(intents_num, full_list):
    final_intents_lists = []
    for set_size in range(2, intents_num + 1):
        final_intents_lists.extend(list(combinations(full_list, set_size)))
    return final_intents_lists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Amazon/Amazon-Website")
    parser.add_argument("--intents_num", type=int, default=5)
    parser.add_argument("--k_size", type=int, default=3)
    parser.add_argument("--ind_method", type=str, default="IndexFlatL2")
    parser.add_argument("--generate_neighbors_dict", type=bool, default=False)
    parser.add_argument("--load_datasets", type=bool, default=True)
    parser.add_argument("--hidden_channels", type=int, default=250)
    parser.add_argument("--epochs_num", type=int, default=150)
    parser.add_argument("--GNN_model", type=str, default='GraphSAGE')
    parser.add_argument("--seeds_num", type=int, default=4)
    parser.add_argument("--ablation", type=bool, default=False)
    parser.add_argument("--GNN_layers", type=int, default=3)

    hp = parser.parse_args()
    task = hp.task
    intents_num = hp.intents_num
    generate_neighbors_dict = hp.generate_neighbors_dict
    k_size = hp.k_size
    ind_method = hp.ind_method
    hidden_channels = hp.hidden_channels
    epochs_num = hp.epochs_num
    load_datasets = hp.load_datasets
    gnn_model = hp.GNN_model
    seeds_num = hp.seeds_num
    ablation = hp.ablation
    gnn_layers = hp.GNN_layers

    file_types = ['train', 'valid', 'test']
    full_intent_list = [intent for intent in range(intents_num)]
    hidden_channels_values = [50*h for h in range(1, 11)]
    k_vals = [2*k for k in range(6)]

    if ablation:
        intent_lists = get_intent_lists(intents_num, full_intent_list)
    else:
        intent_lists = [tuple(full_intent_list)]

    if generate_neighbors_dict:
        for seed in range(seeds_num):
            print(f'Seed {seed}')
            train_dataloader, valid_dataloader, test_dataloader = create_dataloaders_inference(task, full_intent_list,
                                                                                               True,
                                                                                               k_size, file_types,
                                                                                               ind_method, seed,
                                                                                               load_datasets)
    for hidden_channels in hidden_channels_values:
        for k in k_vals:
            print(f'k= {k}:')
            for seed in range(seeds_num):
                print(f'Seed {seed}:')
                for intent_list in intent_lists:
                    if 0 not in intent_list:
                        continue
                    print(f'Intents:{intent_list}')
                    print(3 * '#################')
                    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders_inference(task, intent_list,
                                                                                                       False,
                                                                                                       k, file_types,
                                                                                                       ind_method, seed,
                                                                                                       load_datasets)

                    train_and_evaluate(train_dataloader, valid_dataloader, test_dataloader,
                                       gnn_model, intent_list, hidden_channels,
                                       task, epochs_num, seed, k, ablation, gnn_layers)
