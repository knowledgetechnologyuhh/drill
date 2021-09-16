import logging
import random

import torch
from torch import nn
from transformers import BertTokenizer, BertModel

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BaseModelLog')


class BertBase(nn.Module):
    """
    BERT model with additional downsampling layer used for MTL, SEQ, and REPLAY
    """
    def __init__(self, n_classes, max_length, hidden_dim, device):
        super(BertBase, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.downsampler = nn.Linear(768, hidden_dim)
        self.linear = nn.Linear(hidden_dim, self.n_classes)
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs, out_from='full'):
        if out_from == 'full':
            outputs = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            out = self.downsampler(outputs[1])
            out = self.linear(out)
        elif out_from == 'transformers':
            out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])[1]
        elif out_from == 'downsampler':
            outputs = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            out = self.downsampler(outputs[1])
        elif out_from == 'linear':
            out = self.linear(inputs)
        else:
            raise ValueError('Invalid value of argument')
        return out


class BertPN(nn.Module):
    """
    BERT model used by ANML
    """
    def __init__(self, n_classes, max_length, device):
        super(BertPN, self).__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.to(self.device)
        self.linear = nn.Linear(768, n_classes)
        self.linear.to(device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs, input_b=None, out_from='full'):
        if out_from == 'full':
            input_a = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])[1]
            out = self.linear(input_a * input_b)

        elif out_from == 'linear':
            out = self.linear(inputs * input_b)

        elif out_from == 'transformers':
            out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])[1]

        else:
            raise ValueError('Invalid value of argument')
        return out


class BertNM(nn.Module):
    """
    BERT neuromodulating model (with frozen parameters) used for ANML
    """
    def __init__(self, device):
        super(BertNM, self).__init__()
        self.device = device
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, 768))
        self.to(self.device)

    def forward(self, inputs):
        outputs = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(outputs[1])
        return out


class BertRLN(nn.Module):
    """
    BERT model with downsampling and batchnorm used by OML and DRILL
    """
    def __init__(self, max_length, hidden_dim, device, activate=True):
        super(BertRLN, self).__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        if activate:
            self.downsampler = nn.Sequential(nn.Linear(768, hidden_dim), nn.Tanh())
        else:
            self.downsampler = nn.Sequential(nn.Linear(768, hidden_dim), nn.BatchNorm1d(hidden_dim))

        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        outputs = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.downsampler(outputs[1])
        return out


class LinearPLN(nn.Module):
    """
    PLN with simple linear layer used for OML
    """
    def __init__(self, in_dim, out_dim, device):
        super(LinearPLN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.to(device)

    def forward(self, inputs):
        out = self.linear(inputs)
        return out


class GatingPLN(nn.Module):
    """
    PLN with knowledge integration mechanism used for DRILL
    """
    def __init__(self, in_dim, out_dim, device, fusion):
        super(GatingPLN, self).__init__()
        self.fusion = fusion
        if fusion == "gate":
            self.linear = nn.Linear(in_dim, out_dim)
            self.linear.to(device)
        elif fusion == "cat":
            self.linear_a = nn.Linear(in_dim, in_dim // 2)
            self.linear_a.to(device)
            self.linear_b = nn.Linear(in_dim, in_dim // 2)
            self.linear_b.to(device)
            self.linear = nn.Linear((in_dim // 2) * 2, out_dim)
            self.linear.to(device)
        else:
            raise ValueError('Invalid value of argument')

    def forward(self, input_a, input_b, out_from='full'):
        if self.fusion == "gate":
            fusion = input_a * input_b
            out = self.linear(fusion)
        elif self.fusion == "cat":
            linear_a = self.linear_a(input_a)
            linear_b = self.linear_b(input_b)
            fusion = torch.cat((linear_a, linear_b), dim=1)
            out = self.linear(fusion)
        else:
            raise ValueError('Invalid value of argument')

        if out_from == 'fusion':
            return fusion
        elif out_from == 'full':
            return out
        else:
            raise ValueError('Invalid value of argument')


class EpisodicMemory:
    """
    Replay memory M_E used by ANML, OML, DRILL, and REPLAY
    """

    def __init__(self, tuple_size, write_probs):
        self.buffer = []
        self.tuple_size = tuple_size
        self.write_probs = write_probs

    def write(self, input_tuple, write_prob):
        if random.random() < write_prob:
            self.buffer.append(input_tuple)

    def read(self):
        return random.choice(self.buffer)

    def write_batch(self, *elements, ds_idx=0):
        element_list = []
        for e in elements:
            if isinstance(e, torch.Tensor):
                element_list.append(e.tolist())
            else:
                element_list.append(e)
        for write_tuple in zip(*element_list):
            try:
                self.write(write_tuple, self.write_probs[ds_idx])
            except:
                continue

    def read_batch(self, batch_size):
        contents = [[] for _ in range(self.tuple_size)]
        for _ in range(batch_size):
            read_tuple = self.read()
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        return tuple(contents)

    def len(self):
        return len(self.buffer)

    def reset_memory(self):
        self.buffer = []
