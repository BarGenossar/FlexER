import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import argparse
import json

from torch.utils import data
from .model import MultiTaskNet
from .dataset import *
from .train_util import *
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp


def train(model, train_set, optimizer, inference, intents_num, seed,
          MCML_inference, ML_head, scheduler=None, batch_size=32, fp16=False):
    torch.manual_seed(seed)
    iterator = data.DataLoader(dataset=train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=1,
                               collate_fn=SnippextDataset.pad)

    tagging_criterion = nn.CrossEntropyLoss(ignore_index=0)
    classifier_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    if ML_head:
        log_vars = nn.Parameter(torch.zeros(intents_num + 1))
    else:
        log_vars = nn.Parameter(torch.zeros(intents_num))
    model.train()
    for i, batch in enumerate(iterator):
        # for monitoring
        words, x, is_heads, tags, mask, y, seqlens, taskname = batch
        taskname = taskname[0]
        _y = y

        if 'tagging' in taskname:
            criterion = tagging_criterion
        elif 'sts-b' in taskname:
            criterion = regression_criterion
        else:
            if inference == "Multilabel":
                criterion = nn.BCEWithLogitsLoss()
            elif inference == 'MCML':
                # Multiclass-Multilabel
                criterion_intent = classifier_criterion
                criterion_Multiple = nn.BCEWithLogitsLoss()
            else:
                criterion = classifier_criterion

        # forward
        optimizer.zero_grad()
        logits, y, _ = model(x, y, task=taskname)
        if 'sts-b' in taskname:
            logits = logits.view(-1)
        else:
            if inference == 'MCML' and MCML_inference != 'Multilabel':
                for intent in range(intents_num + 1):
                    logits[intent] = logits[intent].view(-1, logits[intent].shape[-1])
            else:
                logits = logits.view(-1, logits.shape[-1])
        if inference == "Multilabel":
            y = y.type_as(logits)
            loss = criterion(logits, y)
        elif inference == 'MCML':
            loss = 0.0
            for intent in range(intents_num):
                # Run over the binary classifiers
                tmp_precision = torch.exp(-log_vars[intent])
                tmp_y = y[:, intent].view(-1)
                tmp_loss = criterion_intent(logits[intent], tmp_y)
                loss += tmp_precision * tmp_loss + log_vars[intent]

            if ML_head:
                # Multilabel
                tmp_precision = torch.exp(-log_vars[intents_num])
                # tmp_y = y[:, intents_num].type_as(logits[intents_num])
                tmp_y = y.type_as(logits[intents_num])
                tmp_loss = criterion_Multiple(logits[intents_num], tmp_y)
                loss += tmp_precision * tmp_loss + log_vars[intents_num]

        else:
            y = y.view(-1)
            loss = criterion(logits, y)

        # back propagation
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i == 0:
            # print("=====sanity check======")
            # print("words:", words[0])
            # print("x:", x.cpu().numpy()[0][:seqlens[0]])
            # print("tokens:", get_tokenizer().convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            # print("is_heads:", is_heads[0])
            # y_sample = _y.cpu().numpy()[0]
            # if np.isscalar(y_sample):
            #     print("y:", y_sample)
            # else:
            #     print("y:", y_sample[:seqlens[0]])
            # print("tags:", tags[0])
            # print("mask:", mask[0])
            # print("seqlen:", seqlens[0])
            # print("task_name:", taskname)
            print("=======================")

        if i%10 == 0: # monitoring
            print(f"step: {i}, task: {taskname}, loss: {loss.item()}")
            del loss

def initialize_and_train(task_config,
                         trainset,
                         validset,
                         testset,
                         hp,
                         seed,
                         run_tag, task,
                         inference, MCML_inference,
                         intents_num,
                         ML_head=None):
    """The train process.

    Args:
        task_config (dictionary): the configuration of the task
        trainset (SnippextDataset): the training set
        validset (SnippextDataset): the validation set
        testset (SnippextDataset): the testset
        hp (Namespace): the parsed hyperparameters
        run_tag (string): the tag of the run (for logging purpose)

    Returns:
        None
    """
    # create iterators for validation and test
    padder = SnippextDataset.pad
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size * 4,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size * 4,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiTaskNet([task_config], device, inference, hp.finetuning,
                         seed, lm=hp.lm, bert_path=hp.bert_path)
    if device == 'cpu':
        optimizer = AdamW(model.parameters(), lr=hp.lr)
    else:
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=hp.lr)
        if hp.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # learning rate scheduler
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_steps // 10,
                                                num_training_steps=num_steps)

    # create logging directory
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    epoch = 1
    while epoch <= hp.n_epochs:
        train(model,
              trainset,
              optimizer,
              inference, intents_num,
              seed, MCML_inference,
              ML_head,
              scheduler=scheduler,
              batch_size=hp.batch_size,
              fp16=hp.fp16)

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1 = eval_on_task(epoch,
                            model,
                            task_config['name'],
                            valid_iter,
                            validset,
                            test_iter,
                            testset,
                            writer,
                            run_tag,
                            inference, MCML_inference)

        if dev_f1 > 1e-6:
            epoch += 1
            if hp.save_model:
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    torch.save(model.state_dict(), run_tag + task + '.pt')
                # if test_f1 > best_test_f1:
                #     best_test_f1 = test_f1
                #     torch.save(model.state_dict(), run_tag + '_test.pt')

    writer.close()

