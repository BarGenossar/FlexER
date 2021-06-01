from GCN_model import GCN
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process_graph import GraphReaderFlexER
from torch_geometric.data import DataLoader as GeometricDataLoader
import torch_geometric
import csv
import copy
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
import pandas as pd
import torch.nn as nn

torch_geometric.set_debug(True)


def find_device():
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def create_dataloaders_inference(files_path, task, intent, intents_num, emb_type, batch_size, graph_type, seed=1,
                                 inference=None, MCML_inference=None):
    train_datareader = GraphReaderFlexER("train", task, files_path,
                                         intent, intents_num,
                                         emb_type, graph_type=graph_type,
                                         inference=inference, MCML_inference=MCML_inference)
    correlations = train_datareader.get_correlations
    valid_dataloader = GraphReaderFlexER("valid", task, files_path,
                                         intent, intents_num,
                                         emb_type,
                                         correlations,
                                         graph_type=graph_type,
                                         inference=inference,
                                         MCML_inference=MCML_inference)
    test_datareader = GraphReaderFlexER("test", task, files_path,
                                        intent, intents_num,
                                        emb_type,
                                        correlations,
                                        graph_type=graph_type,
                                        inference=inference,
                                        MCML_inference=MCML_inference)
    torch.manual_seed(seed)
    train_dataloader = GeometricDataLoader(train_datareader, batch_size=batch_size, shuffle=True)
    valid_dataloader = GeometricDataLoader(valid_dataloader, batch_size=batch_size, shuffle=False)
    test_dataloader = GeometricDataLoader(test_datareader, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader


def epoch_train(model, train_dataloader, criterion, optimizer, device, intent, intents_num, scheduler, batch_size, seed=1):
    model.train()
    torch.manual_seed(seed)
    for batch_data in train_dataloader:  # Iterate in batches over the training dataset
        data_items = batch_data
        x = data_items.x.to(device)
        labels = data_items.y.to(device)
        edges = data_items.edge_index.to(device)
        data_items.batch.to(device)
        relevant_batch_indices = [intent + i*intents_num for i in range(x.shape[0] // intents_num)]
        # print("x shape: ", x.shape)
        # print("relevant_batch_indices: ", relevant_batch_indices)
        model.zero_grad()
        out, softmax_vals = model(x, edges, relevant_batch_indices, labels)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        del x, labels, edges
        torch.cuda.empty_cache()
    return model


def epoch_test(model, rel_dataloader, criterion, device, intent, intents_num, seed=1):
    model.eval()
    correct_preds = 0.0
    total_loss = 0.0
    predictions_list = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for batch_data in rel_dataloader:
            data_items = batch_data
            x = data_items.x.to(device)
            labels = data_items.y.to(device)
            edges = data_items.edge_index.to(device)
            data_items.batch.to(device)
            relevant_batch_indices = [intent + i*intents_num for i in range(x.shape[0] // intents_num)]
            out, softmax_vals = model(x, edges, relevant_batch_indices, labels)
            total_loss += criterion(out, labels)
            preds = torch.argmax(out, dim=1)
            correct_preds += int((preds == labels).sum())
            predictions_list.extend(preds.tolist())
            del x, labels, edges
            torch.cuda.empty_cache()
    return correct_preds / len(rel_dataloader.dataset), total_loss / len(rel_dataloader), predictions_list


def get_features_num(emb_type, inference=None, MCML_inference=None):
    if emb_type == 'probs':
        return 2
    elif inference == 'MCML' and MCML_inference != 'Multilabel':
        return 384
    else:
        return 768


def update_predictions_csv(predictions_list, task, intent, emb_type, files_path="output/"):
    files_path += task + '/'
    clean_task = task.split('/')[1]
    intent = str(intent)
    file_types = ['train', 'valid', 'test']
    for file_type, best_preds in zip(file_types, predictions_list):
        if intent == '0':
            df = pd.read_csv(files_path + clean_task + '_' + file_type + '_' + emb_type + '.csv')
        else:
            df = pd.read_csv(files_path + clean_task + '_' + file_type + '_' + emb_type + '_final.csv')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['GCN' + intent] = df['index'].apply(lambda ind: best_preds[ind])
        df.to_csv(files_path + clean_task + '_' + file_type + '_' + emb_type + '_final.csv')
    return


def calc_F1_final(task, intent, emb_type, files_path="output/"):
    files_path += task + '/'
    clean_task = task.split('/')[1]
    intent = str(intent)
    file_types = ['train', 'valid', 'test']
    for file_type in file_types:
        df = pd.read_csv(files_path + clean_task + '_' + file_type + '_' + emb_type + '_final.csv')
        labels = df['label' + intent]
        predictions_base = df['prediction' + intent]
        predictions_GCN = df['GCN' + intent]
        acc_base = (labels == predictions_base).sum() / len(labels)
        acc_GCN = (labels == predictions_GCN).sum() / len(labels)
        rec_base = ((labels == 1) & (labels == predictions_base)).sum() / (labels == 1).sum()
        rec_GCN = ((labels == 1) & (labels == predictions_GCN)).sum() / (labels == 1).sum()
        prec_base = ((predictions_base == 1) & (labels == predictions_base)).sum() / (predictions_base == 1).sum()
        prec_GCN = ((predictions_GCN == 1) & (labels == predictions_GCN)).sum() / (predictions_GCN == 1).sum()
        F1_base = (2 * prec_base * rec_base) / (prec_base + rec_base)
        F1_GCN = (2 * prec_GCN * rec_GCN) / (prec_GCN + rec_GCN)
        print()
        print('File type: ', file_type)
        print(f'base F1: {F1_base:.4f}, base Acc: {acc_base:.4f}, base Prec: {prec_base:.4f}, base Rec: {rec_base:.4f}')
        print(f'GCN F1: {F1_GCN:.4f}, GCN Acc: {acc_GCN:.4f}, GCN Prec: {prec_GCN:.4f}, GCN Rec: {rec_GCN:.4f}')
        print(40 * '==')
        print()
    return


def calc_F1(task, intent, emb_type, predictions_list, file_type, files_path="output/"):
    clean_task = task.split('/')[1]
    files_path += task + '/'
    intent = str(intent)
    df = pd.read_csv(files_path + clean_task + '_' + file_type + '_' + emb_type + '.csv')
    labels = df['label' + intent]
    predictions = pd.Series(predictions_list)
    rec_GCN = ((labels == 1) & (labels == predictions)).sum() / (labels == 1).sum()
    prec_GCN = ((predictions == 1) & (labels == predictions)).sum() / (predictions == 1).sum()
    F1_GCN = (2 * prec_GCN * rec_GCN) / (prec_GCN + rec_GCN)
    return F1_GCN


def run_graph_classification(files_path, task, intent,
                             intents_num,
                             hidden_channels,
                             epochs_num, emb_type,
                             batch_size, graph_type,
                             inference, MCML_inference):
    torch.manual_seed(1)
    device = find_device()
    features_num = get_features_num(emb_type, inference, MCML_inference)
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders_inference(files_path,
                                                                                       task, intent,
                                                                                       intents_num,
                                                                                       emb_type,
                                                                                       batch_size,
                                                                                       graph_type, seed=1,
                                                                                       inference=inference,
                                                                                       MCML_inference=MCML_inference)

    model = GCN(features_num, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()
    total_steps = int(len(train_dataloader) * epochs_num)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.to(device)
    best_valid_F1 = 0.0
    best_test_F1 = 0.0
    torch.manual_seed(1)
    for epoch in range(1, epochs_num + 1):
        model = epoch_train(model, train_dataloader, criterion, optimizer,
                            device, intent, intents_num, scheduler, batch_size)
        train_acc, train_loss, predictions_train = \
            epoch_test(model, train_dataloader, criterion, device, intent, intents_num)
        valid_acc, valid_loss, predictions_valid = \
            epoch_test(model, valid_dataloader, criterion, device, intent, intents_num)
        test_acc, test_loss, predictions_test = \
            epoch_test(model, test_dataloader, criterion, device, intent, intents_num)
        F1_train = calc_F1(task, intent, emb_type, predictions_train, 'train', files_path="output/")
        F1_valid = calc_F1(task, intent, emb_type, predictions_valid, 'valid', files_path="output/")
        F1_test = calc_F1(task, intent, emb_type, predictions_test, 'test', files_path="output/")
        print(f'Epoch: {epoch:03d}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}')
        print(f'Train F1: {F1_train:.4f}, Val F1: {F1_valid:.4f}, Test F1: {F1_test:.4f}')
        print()
        if F1_valid > best_valid_F1:
            best_valid_F1 = F1_valid
            # TODO: save model
            best_preds_train = predictions_train.copy()
            best_preds_valid = predictions_valid .copy()
            best_preds_test = predictions_test.copy()
        if F1_test > best_test_F1:
            best_test_F1 = F1_test
            # TODO: save model
    predictions_list = [best_preds_train, best_preds_valid, best_preds_test]
    update_predictions_csv(predictions_list, task, intent, emb_type)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='WDC/WDC_small')
    parser.add_argument("--files_path", type=str, default='data/')
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--intent", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--intents_num", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=2048)
    parser.add_argument("--emb_type", type=str, default='poolers')
    parser.add_argument("--calc_F1", type=bool, default=False)
    parser.add_argument("--graph_type", type=str, default='complete')
    parser.add_argument("--inference", type=str, default=None)
    parser.add_argument("--MCML_inference", type=str, default="Multilabel")

    hp = parser.parse_args()

    torch.manual_seed(1)
    if hp.calc_F1:
        calc_F1_final(hp.task, hp.intent, hp.emb_type)
    else:
        run_graph_classification(hp.files_path, hp.task, hp.intent,
                                 hp.intents_num,
                                 hp.hidden_channels,
                                 hp.n_epochs, hp.emb_type,
                                 hp.batch_size,
                                 hp.graph_type,
                                 hp.inference,
                                 hp.MCML_inference)
        calc_F1_final(hp.task, hp.intent, hp.emb_type)
