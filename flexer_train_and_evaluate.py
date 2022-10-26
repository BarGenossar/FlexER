from GNN_models import GraphSAGE2, GraphSAGE3
from torch_geometric.nn import to_hetero
import pandas as pd
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model_input(input_data, intent, device):
    input_data.to(device)
    x_input = input_data[''.join(('intent', str(intent)))].x
    labels = input_data[''.join(('intent', str(intent)))].y.type(torch.LongTensor)
    edges = input_data.edge_index_dict
    return x_input, labels, edges


def update_results_dict(results_dict, intent, preds, labels):
    preds = list(preds)
    labels = list(labels)
    results_dict[intent] = {'preds': preds, 'labels': labels}
    return results_dict


def create_data_path_out(task, seed):
    if 'WDC' in task or 'Amazon-Website' in task:
        return ''.join(("data/", task, "/", str(seed), "/", task.split('/')[1]))
    elif "Structured" in task:
        return ''.join(("data/er_magellan/", task, "/", str(seed), "/", task.split('/')[1]))
    else:
        raise ValueError("Data path is not valid!")


def modify_dict(results_dict):
    pooler_ids = [pooler_id for pooler_id in range(len(results_dict[min(results_dict.keys())]['labels']))]
    preds_list = [[results_dict[intent]['preds'][pooler_id].item()
                   for pooler_id in pooler_ids] for intent in results_dict.keys()]
    labels_list = [[results_dict[intent]['labels'][pooler_id].item()
                    for pooler_id in pooler_ids] for intent in results_dict.keys()]
    cols_input = tuple(preds_list + labels_list)
    return dict(zip(pooler_ids, zip(*cols_input)))


def create_results_path(task, gnn_model, seed, hidden_channels, k_size, intents_list, ablation, gnn_layers):
    if ablation:
        str_intents_list = [str(intent) for intent in intents_list]
        str_intents = ''.join(str_intents_list)
        results_path = ''.join(('results/', task, '/', gnn_model, '/', str(gnn_layers), 'Layers/Ablation/',
                                str_intents, '/', str(hidden_channels), '/k=',
                                str(k_size), '/', str(seed), '/'))

    else:
        results_path = ''.join(('results/', task, '/', gnn_model, '/', str(gnn_layers), 'Layers/',
                                str(hidden_channels), '/k=', str(k_size),
                                '/', str(seed), '/'))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path


def create_results_df(results_dict, task, seed, gnn_model, hidden_channels, k_size, intents_list, ablation, gnn_layers):
    results_path = create_results_path(task, gnn_model, seed, hidden_channels, k_size, intents_list, ablation, gnn_layers)
    modified_dict = modify_dict(results_dict)
    df = pd.DataFrame.from_dict(modified_dict, orient='index')
    df.reset_index(inplace=True)
    preds_cols = [''.join(('preds', str(intent))) for intent in results_dict.keys()]
    labels_cols = [''.join(('labels', str(intent))) for intent in results_dict.keys()]
    df.columns = ['pooler_id'] + preds_cols + labels_cols
    df.to_csv(''.join((results_path, 'results.csv')))
    return


def train_model(model, input_data, optimizer, criterion, intent, seed):
    torch.manual_seed(seed)
    model.train()
    labels = input_data[''.join(('intent', str(intent)))].y.type(torch.LongTensor).to(device)
    optimizer.zero_grad()
    out = model(input_data.x_dict, input_data.edge_index_dict)[''.join(('intent', str(intent)))]
    preds = out.argmax(dim=1)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return calc_F1(preds, labels)


def eval_model(model, input_data, intent, seed):
    torch.manual_seed(seed)
    model.eval()
    labels = input_data[''.join(('intent', str(intent)))].y.type(torch.LongTensor).to(device)
    out = model(input_data.x_dict, input_data.edge_index_dict)[''.join(('intent', str(intent)))]
    preds = out.argmax(dim=1)
    return calc_F1(preds, labels), preds, labels


def calc_F1(preds, labels):
    if (preds == 1).sum().item() == 0:
        return torch.tensor([0])
    rec = ((labels == 1) & (preds == labels)).sum() / (labels == 1).sum()
    prec = ((preds == 1) & (preds == labels)).sum() / (preds == 1).sum()
    return (2 * prec * rec) / (prec + rec)


def get_model(gnn_model, hidden_channels, seed, gnn_layers):
    if gnn_layers == 2:
        return GraphSAGE2(hidden_channels, seed)
    else:
        return GraphSAGE3(hidden_channels, seed)



def train_and_evaluate(train_dataloader, valid_dataloader, test_dataloader,
                       gnn_model, intents_list, hidden_channels,
                       task, epochs_num, seed, k_size, ablation, gnn_layers):
    model = get_model(gnn_model, hidden_channels, seed, gnn_layers)
    model = to_hetero(model, train_dataloader.metadata(), aggr='mean')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    results_dict = {intent: dict() for intent in intents_list}

    train_dataloader.to(device)
    valid_dataloader.to(device)
    test_dataloader.to(device)

    for intent in intents_list:
        best_valid_F1 = 0.0
        final_f1_test = 0.0
        print(f'Intent {intent}:')
        for epoch in range(1, epochs_num + 1):
            F1_train = train_model(model, train_dataloader, optimizer, criterion, intent, seed)
            F1_valid, _, _ = eval_model(model, valid_dataloader, intent, seed)
            F1_test, preds_test, labels_test = eval_model(model, test_dataloader, intent, seed)
            if epoch % 5 == 0:
                print(f'Epoch {epoch} Results- '
                      f'Train F1: {F1_train.item():.4f}, '
                      f'Val F1: {F1_valid.item():.4f}, '
                      f'Test F1: {F1_test.item():.4f}')
            if best_valid_F1 < F1_valid.item():
                results_dict = update_results_dict(results_dict, intent, preds_test, labels_test)
                best_valid_F1 = F1_valid
                final_f1_test = F1_test
        print(f'F1 Test: {round(final_f1_test.item(), 3)}')
        print(3*'========================')
    print(5 * '************************************')
    create_results_df(results_dict, task, seed, gnn_model, hidden_channels, k_size, intents_list, ablation, gnn_layers)
    return
