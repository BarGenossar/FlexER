import torch
import torch.nn as nn
import os
import numpy as np
import random
import json
import jsonlines
import csv
import re
import time
import argparse
import sys
import traceback

from torch.utils import data
from tqdm import tqdm
from apex import amp
from scipy.special import softmax

sys.path.insert(0, "Snippext_public")
from snippext.model import MultiTaskNet
from ditto.exceptions import ModelNotFoundError
from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *

def to_str(row, summarizer=None, max_len=256, dk_injector=None):
    """Serialize a data entry

    Args:
        row (Dictionary): the data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        string: the serialized version
    """
    # if the entry is already serialized
    if isinstance(row, str):
        return row
    content = ''
    for attr in row.keys():
        content += 'COL %s VAL %s ' % (attr, row[attr])

    if summarizer is not None:
        content = summarizer.transform(content, max_len=max_len)

    if dk_injector is not None:
        content = dk_injector.transform(content)

    return content

def classify(sentence_pairs, config, model, inference=None, lm='distilbert', max_len=256):
    """Apply the MRPC model.

    Args:
        sentence_pairs (list of tuples of str): the sentence pairs
        config (dict): the model configuration
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length

    Returns:
        list of float: the scores of the pairs
    """
    inputs = []
    for (sentA, sentB) in sentence_pairs:
        inputs.append(sentA + '\t' + sentB)

    dataset = DittoDataset(inputs, config['vocab'], config['name'], lm=lm, max_len=max_len)
    iterator = data.DataLoader(dataset=dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=DittoDataset.pad)

    # prediction
    Y_logits = []
    Y_poolers = []
    Y_hat = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            logits, _, y_hat, poolers = model(x, y, task=taskname, get_enc=True)  # y_hat: (N, T)
            intents_num = model.get_intents_num()
            if inference == 'Multilabel':
                Y_logits.extend(logits.cpu())
            elif inference == 'MCML':
                if MCML_inference != 'Multilabel':
                    Y_logits.extend(logits[int(MCML_inference)].cpu())
                    poolers = poolers[int(MCML_inference)]
                    y_hat = y_hat[int(MCML_inference)]
                else:
                    Y_logits.extend(logits[intents_num].cpu())
            else:
                Y_logits += logits.cpu().numpy().tolist()
            poolers = poolers.cpu().numpy().tolist()
            poolers = [[round(elem, 4) for elem in tensor] for tensor in poolers]
            Y_poolers += poolers
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    results = []
    for i in range(len(inputs)):
        if inference == 'Multilabel':
            softmax = torch.sigmoid(Y_logits[i])
            pred = (softmax > 0.5).int().cpu().tolist()
            results.append(pred)
        elif inference == 'MCML':
            if MCML_inference != 'Multilabel':
                pred = dataset.idx2tag[Y_hat[i]]
                results.append(pred)
        else:
            pred = dataset.idx2tag[Y_hat[i]]
            results.append(pred)
    return results, Y_logits, Y_poolers


def predict(input_path, output_path, config, model,
            batch_size=1024,
            summarizer=None,
            lm='distilbert',
            max_len=256,
            dk_injector=None,
            inference=None,
            MCML_inference=None):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        input_path (str): the input file path
        output_path (str): the output file path
        config (Dictionary): the task configuration
        model (SnippextModel): the model for prediction
        intent (int): the granular intent number
        batch_size (int): the batch size
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        None
    """
    pairs = []

    def process_batch(rows, pairs, writer):
        try:
            predictions, logits, poolers = classify(pairs, config, model, inference, lm=lm, max_len=max_len)
        except:
            # ignore the whole batch
            return
        if inference == 'Multilabel':
            scores = [torch.sigmoid(logit) for logit in logits]
        elif inference == 'MCML':
            if MCML_inference != 'Multilabel':
                scores = [softmax(logit) for logit in logits]
                # scores = softmax(logits, axis=1)
            else:
                scores = [torch.sigmoid(logit) for logit in logits]
        else:
            scores = softmax(logits, axis=1)
        for row, pred, score, pooler in zip(rows, predictions, scores, poolers):
            if inference == 'Multilabel':
                match_confidence = score.tolist()
            elif inference == 'MCML':
                if inference != 'Multilabel':
                    match_confidence = round(score[int(pred)].item(), 4)
                else:
                    match_confidence = score.tolist()
            else:
                match_confidence = round(score[int(pred)], 4)
            output = {'left': row[0], 'right': row[1],
                      'prediction': pred,
                      'match_confidence': match_confidence,
                      'pooler': pooler}
            writer.write(output)

    # input_path can also be train/valid/test.txt
    # convert to jsonlines
    # input_path = input_path.replace('.txt', str(intent) + ".txt")
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    # batch processing
    start_time = time.time()
    with jsonlines.open(input_path) as reader,\
         jsonlines.open(output_path, mode='w') as writer:
        pairs = []
        rows = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append((to_str(row[0], summarizer, max_len, dk_injector),
                          to_str(row[1], summarizer, max_len, dk_injector)))
            rows.append(row)
            if len(pairs) == batch_size:
                process_batch(rows, pairs, writer)
                pairs.clear()
                rows.clear()

        if len(pairs) > 0:
            process_batch(rows, pairs, writer)

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s_dk=%s_su=%s' % (config['name'], lm, str(dk_injector != None), str(summarizer != None))
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))

def load_model(task, path, lm, use_gpu, inference=None, fp16=True):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """
    # load models
    model_name = task.split('/')[1]
    if inference is None:
        full_path = task[:-1] + '/' + model_name
    else:
        full_path = task.split('_')[0] + '/' + model_name
    checkpoint = os.path.join(path, '%s.pt' % full_path)
    if not os.path.exists(checkpoint):
        raise ModelNotFoundError(checkpoint)

    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}

    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    config = configs[task]
    config_list = [config]
    model = MultiTaskNet([config], device, inference, True, lm=lm)

    saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state)


    model = model.to(device)

    if fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return config, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='Structured/Beer')
    parser.add_argument("--input_path", type=str, default='input/candidates_small.jsonl')
    parser.add_argument("--output_path", type=str, default='output/matched_small.jsonl')
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/')
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--intent", type=int, default=0)
    parser.add_argument("--intents_num", type=int, default=2)
    parser.add_argument("--inference", type=str, default=None)
    parser.add_argument("--MCML_inference", type=str, default="Multilabel")
    hp = parser.parse_args()

    main_task = hp.task
    file_types = ['train', 'valid', 'test']
    inference = hp.inference
    MCML_inference = hp.MCML_inference

    if inference is None:
        for intent in range(hp.intents_num):
            for file_type in file_types:
                print("********************" + file_type + str(intent) + "********************")
                # create the tag of the run
                task = main_task + str(intent)
                task_name = task.split('/')[1]
                input_path = hp.input_path + '/' + task_name[:-1] + '_' + file_type + str(intent) + '.txt'
                output_path = hp.input_path + '/' + task_name[:-1] + '_' + file_type + str(intent) + '_output.txt'
                # load the models
                config, model = load_model(task, hp.checkpoint_path,
                                           hp.lm, hp.use_gpu, inference, hp.fp16)


                summarizer = dk_injector = None
                if hp.summarize:
                    summarizer = Summarizer(config, hp.lm)

                if hp.dk is not None:
                    if 'product' in hp.dk:
                        dk_injector = ProductDKInjector(config, hp.dk)
                    else:
                        dk_injector = GeneralDKInjector(config, hp.dk)

                # run prediction
                predict(input_path, output_path, config, model,
                        summarizer=summarizer,
                        max_len=hp.max_len,
                        lm=hp.lm,
                        dk_injector=dk_injector,
                        inference=inference)
    elif inference == 'Multilabel':
        for file_type in file_types:
            task = main_task + '_Multilabel'
            # task = main_task
            task_name = task.split('/')[1]
            input_path = hp.input_path + '/' + task_name.split('_')[0] + '_' + file_type + '_Multilabel.txt'
            output_path = hp.input_path + '/' + task_name.split('_')[0] + '_' + file_type + '_Multilabel_output.txt'
            # load the models
            config, model = load_model(task, hp.checkpoint_path,
                                       hp.lm, hp.use_gpu, inference, hp.fp16)
            summarizer = dk_injector = None
            if hp.summarize:
                summarizer = Summarizer(config, hp.lm)

            if hp.dk is not None:
                if 'product' in hp.dk:
                    dk_injector = ProductDKInjector(config, hp.dk)
                else:
                    dk_injector = GeneralDKInjector(config, hp.dk)

            predict(input_path, output_path, config, model,
                    summarizer=summarizer,
                    max_len=hp.max_len,
                    lm=hp.lm,
                    dk_injector=dk_injector,
                    inference=inference)

    elif inference == 'MCML':
        for file_type in file_types:
            task = main_task + '_MCML'
            # task = main_task
            task_name = task.split('/')[1]
            input_path = hp.input_path + '/' + task_name.split('_')[0] + '_' + file_type + '_MCML.txt'
            output_path = hp.input_path + '/' + task_name.split('_')[0] + '_'\
                          + file_type + '_' + MCML_inference + '_MCML_output.txt'
            # load the models
            config, model = load_model(task, hp.checkpoint_path,
                                       hp.lm, hp.use_gpu, inference, hp.fp16)
            summarizer = dk_injector = None
            if hp.summarize:
                summarizer = Summarizer(config, hp.lm)

            if hp.dk is not None:
                if 'product' in hp.dk:
                    dk_injector = ProductDKInjector(config, hp.dk)
                else:
                    dk_injector = GeneralDKInjector(config, hp.dk)

            predict(input_path, output_path, config, model,
                    summarizer=summarizer,
                    max_len=hp.max_len,
                    lm=hp.lm,
                    dk_injector=dk_injector,
                    inference=inference,
                    MCML_inference=MCML_inference)

