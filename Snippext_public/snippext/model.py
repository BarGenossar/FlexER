import torch
import torch.nn as nn
from transformers import BertModel, AlbertModel, DistilBertModel, RobertaModel, XLNetModel, LongformerModel

model_ckpts = {'bert': "bert-base-uncased",
               'albert': "albert-base-v2",
               'roberta': "roberta-base",
               'xlnet': "xlnet-base-cased",
               'distilbert': "distilbert-base-uncased",
               'longformer': "allenai/longformer-base-4096"}

class MultiTaskNet(nn.Module):
    def __init__(self, task_configs=[],
                 device='cpu',
                 inference=None,
                 finetuning=True,
                 seed=1,
                 lm='bert',
                 bert_pt=None,
                 bert_path=None):
        super().__init__()
        torch.manual_seed(seed)
        assert len(task_configs) > 0

        # load the model or model checkpoint
        if bert_path == None:
            if lm == 'bert':
                self.bert = BertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'distilbert':
                self.bert = DistilBertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'albert':
                self.bert = AlbertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'xlnet':
                self.bert = XLNetModel.from_pretrained(model_ckpts[lm])
            elif lm == 'roberta':
                self.bert = RobertaModel.from_pretrained(model_ckpts[lm])
            elif lm == 'longformer':
                self.bert = RobertaModel.from_pretrained(model_ckpts[lm])
        else:
            output_model_file = bert_path
            model_state_dict = torch.load(output_model_file,
                                          map_location=lambda storage, loc: storage)
            if lm == 'bert':
                self.bert = BertModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'distilbert':
                self.bert = DistilBertModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'albert':
                self.bert = AlbertModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'xlnet':
                self.bert = XLNetModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'roberta':
                self.bert = RobertaModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)

        self.device = device
        self.finetuning = finetuning
        self.task_configs = task_configs
        self.module_dict = nn.ModuleDict({})
        self.lm = lm
        self.inference = inference

        # hard corded for now
        hidden_size = 768
        hidden_dropout_prob = 0.1

        for config in task_configs:
            name = config['name']
            task_type = config['task_type']
            vocab = config['vocab']

            if task_type == 'tagging':
                # for tagging
                vocab_size = len(vocab) # 'O' and '<PAD>'
                if 'O' not in vocab:
                    vocab_size += 1
                if '<PAD>' not in vocab:
                    vocab_size += 1
            else:
                # for pairing and classification
                vocab_size = len(vocab)

            self.intents_num = vocab_size

            self.module_dict['%s_dropout' % name] = nn.Dropout(hidden_dropout_prob)
            self.module_dict['%s_fc' % name] = nn.Linear(hidden_size, vocab_size)

            self.vocab_size = vocab_size

            if inference == 'MCML':
                # Multiclass-Multilabel
                for fc_num in range(vocab_size):
                    first_layer = "FC" + str(fc_num) + "_L1"
                    second_layer = "FC" + str(fc_num) + "_L2"
                    droput_name = str(fc_num) + '_dropout'
                    self.module_dict['%s' % first_layer] = nn.Linear(hidden_size, hidden_size//2)
                    self.module_dict['%s' % droput_name] = nn.Dropout(hidden_dropout_prob)
                    self.module_dict['%s' % second_layer] = nn.Linear(hidden_size//2, 2)

    def get_intents_num(self):
        return self.intents_num

    def forward(self, x, y,
                augment_batch=None,
                aug_enc=None,
                second_batch=None,
                x_enc=None,
                task='hotel_tagging',
                get_enc=False):
        """Forward function of the BERT models for classification/tagging.

        Args:
            x (Tensor):
            y (Tensor):
            augment_batch (tuple of Tensor, optional):
            aug_enc (Tensor, optional):
            second_batch (Tensor, optional):
            task (string, optional):
            get_enc (boolean, optional):

        Returns:
            Tensor: logits
            Tensor: y
            Tensor: yhat
            Tensor (optional): enc"""

        # move input to GPU
        x = x.to(self.device)
        y = y.to(self.device)
        if second_batch != None:
            index, lam = second_batch
            lam = torch.tensor(lam).to(self.device)
        if augment_batch != None:
            aug_x, aug_lam = augment_batch
            aug_x = aug_x.to(self.device)
            aug_lam = torch.tensor(aug_lam).to(self.device)

        dropout = self.module_dict[task + '_dropout']
        fc = self.module_dict[task + '_fc']

        if 'tagging' in task: # TODO: this needs to be changed later
            if self.training and self.finetuning:
                self.bert.train()
                if x_enc is None:
                    enc = self.bert(x)[0]
                else:
                    enc = x_enc
                # Dropout
                enc = dropout(enc)
            else:
                self.bert.eval()
                with torch.no_grad():
                    enc = self.bert(x)[0]

            if augment_batch != None:
                if aug_enc is None:
                    aug_enc = self.bert(aug_x)[0]
                enc[:aug_x.shape[0]] *= aug_lam
                enc[:aug_x.shape[0]] += aug_enc * (1 - aug_lam)

            if second_batch != None:
                enc = enc * lam + enc[index] * (1 - lam)
                enc = dropout(enc)

            logits = fc(enc)
            y_hat = logits.argmax(-1)
            if get_enc:
                return logits, y, y_hat, enc
            else:
                return logits, y, y_hat
        else:
            if self.training and self.finetuning:
                self.bert.train()
                if x_enc is None:
                    output = self.bert(x)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = dropout(pooled_output)
                else:
                    pooled_output = x_enc
            else:
                self.bert.eval()
                with torch.no_grad():
                    output = self.bert(x)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = dropout(pooled_output)

            if augment_batch != None:
                if aug_enc is None:
                    aug_enc = self.bert(aug_x)[0][:, 0, :]
                pooled_output[:aug_x.shape[0]] *= aug_lam
                pooled_output[:aug_x.shape[0]] += aug_enc * (1 - aug_lam)

            if second_batch != None:
                pooled_output = pooled_output * lam + pooled_output[index] * (1 - lam)

            logits = fc(pooled_output)
            if 'sts-b' in task:
                y_hat = logits
            else:
                if self.inference == 'Multilabel':
                    softmax = torch.sigmoid(logits)
                    y_hat = (softmax > 0.5).int()
                elif self.inference == 'MCML':
                    y_hat_dict = dict()
                    logits_dict = dict()
                    pooled_output_dict = dict()


                    softmax = torch.sigmoid(logits)
                    # The next row defines the prediction of multiple intents head
                    y_hat_dict[self.vocab_size] = (softmax > 0.5).int()
                    logits_dict[self.vocab_size] = logits
                    pooled_output_dict[self.vocab_size] = pooled_output

                    for intent_num in range(self.vocab_size):
                        fc1 = self.module_dict["FC" + str(intent_num) + "_L1"]
                        current_dropout = self.module_dict[str(intent_num) + '_dropout']
                        fc2 = self.module_dict["FC" + str(intent_num) + "_L2"]
                        tmp_output = fc1(pooled_output)
                        tmp_output = current_dropout(tmp_output)
                        tmp_logits = fc2(tmp_output)
                        # print("tmp_logits: ", tmp_logits)
                        # print("pred: ", tmp_logits.argmax(-1))
                        y_hat_dict[intent_num] = tmp_logits.argmax(-1)
                        logits_dict[intent_num] = tmp_logits
                        pooled_output_dict[intent_num] = tmp_output
                    if get_enc:
                        return logits_dict, y, y_hat_dict, pooled_output_dict
                    else:
                        return logits_dict, y, y_hat_dict
                else:
                    y_hat = logits.argmax(-1)
            if get_enc:
                return logits, y, y_hat, pooled_output
            else:
                return logits, y, y_hat
