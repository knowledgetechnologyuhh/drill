import logging
import math

import higher
import torch
from torch import nn, optim

import numpy as np
import pandas as pd

from torch.utils import data
from transformers import AdamW

import models.utils
from models.base_models import BertRLN, LinearPLN, EpisodicMemory
from datasets.utils import ConcatDataset

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OML-Log')


class OML:
    def __init__(self, device, n_classes, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_probs = kwargs.get('write_probs')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.hidden_dim = kwargs.get('dim')
        self.device = device
        self.online_logger = kwargs.get('online_logger')
        self.rln = BertRLN(kwargs.get('max_len'), self.hidden_dim, device)
        self.pln = LinearPLN(in_dim=self.hidden_dim, out_dim=n_classes, device=device)
        self.memory = EpisodicMemory(tuple_size=2, write_probs=self.write_probs)
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))

        meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                      [p for p in self.pln.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def evaluate(self, dataloader, updates, batch_size):

        self.rln.eval()
        self.pln.train()

        support_set = []
        for _ in range(updates):
            text, labels = self.memory.read_batch(batch_size=batch_size)
            support_set.append((text, labels))

        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpln, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.rln.encode_text(text)
                repr = self.rln(input_dict)
                output = fpln(repr)
                loss = self.loss_fn(output, labels)
                diffopt.step(loss)
                with torch.no_grad():
                    if output.detach().size(1) == 1:
                        pred = (output > 0).int()
                    else:
                        pred = output.max(-1)[1]
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(labels.tolist())

            acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

            logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                        'recall = {:.4f}, F1 score = {:.4f}'.format(np.mean(support_loss), acc, prec, rec, f1))
            if self.online_logger:
                self.online_logger.log_metrics({"loss": np.mean(support_loss), "accuracy" : acc, "precision": prec,
                                                "recall": rec, "f1_score": f1}, prefix="eval_supp_")

            all_losses, all_predictions, all_labels = [], [], []

            for text, labels in dataloader:
                labels = torch.LongTensor(labels).to(self.device)
                input_dict = self.rln.encode_text(text)
                with torch.no_grad():
                    repr = self.rln(input_dict)
                    output = fpln(repr)
                    loss = self.loss_fn(output, labels)
                loss = loss.item()
                with torch.no_grad():
                    if output.detach().size(1) == 1:
                        pred = (output > 0).int()
                    else:
                        pred = output.max(-1)[1]
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))
        if self.online_logger:
            self.online_logger.log_metrics({"loss": np.mean(all_losses), "accuracy": acc, "precision": prec,
                                            "recall": rec, "f1_score": f1}, prefix="eval_")

        return acc, prec, rec, f1

    def training(self, train_datasets, **kwargs):
        updates = kwargs.get('updates')
        batch_size = kwargs.get('batch_size')
        idx_pointer = 0

        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // batch_size
            # Formula for R_F
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 1)))
            # Number of batches appended to query set
            replay_steps = int(self.replay_every * self.replay_rate / batch_size)
        else:
            replay_freq = 0
            replay_steps = 0
        logger.info('Replay frequency: {}'.format(replay_freq))
        logger.info('Replay steps: {}'.format(replay_steps))

        concat_dataset = ConcatDataset(train_datasets)
        train_dataloader = iter(data.DataLoader(concat_dataset, batch_size=batch_size, shuffle=False))

        episode_id, it = 0, 0
        while True:

            self.inner_optimizer.zero_grad()
            support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []

            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpln, diffopt):

                # Inner loop
                support_set = []
                task_predictions, task_labels = [], []

                # Construct support set
                for _ in range(updates):
                    try:
                        text, labels = next(train_dataloader)
                        support_set.append((text, labels))
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return concat_dataset

                # Train on support set
                for text, labels in support_set:
                    labels = torch.LongTensor(labels).to(self.device)
                    input_dict = self.rln.encode_text(text)
                    repr = self.rln(input_dict)
                    output = fpln(repr)
                    loss = self.loss_fn(output, labels)
                    diffopt.step(loss)
                    with torch.no_grad():
                        if output.detach().size(1) == 1:
                            pred = (output > 0).int()
                        else:
                            pred = output.max(-1)[1]
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(labels.tolist())
                    idx_pointer += batch_size
                    self.memory.write_batch(text, labels, ds_idx=concat_dataset.get_idxs(idx_pointer)[0])
                    it += 1

                acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

                logger.info('Episode {} support set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
                                                                        np.mean(support_loss), acc, prec, rec, f1))
                if self.online_logger:
                    self.online_logger.log_metrics({"loss": np.mean(support_loss), "accuracy": acc, "precision": prec,
                                                    "recall": rec, "f1_score": f1}, prefix="train_supp_",
                                                   step=episode_id+1)

                # Outer loop
                query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                query_set = []

                if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0:
                    for _ in range(replay_steps):
                        text, labels = self.memory.read_batch(batch_size=batch_size)
                        query_set.append((text, labels))
                else:
                    try:
                        text, labels = next(train_dataloader)
                        query_set.append((text, labels))
                        idx_pointer += batch_size
                        self.memory.write_batch(text, labels, ds_idx=concat_dataset.get_idxs(idx_pointer)[0])
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return concat_dataset

                for text, labels in query_set:
                    labels = torch.LongTensor(labels).to(self.device)
                    input_dict = self.rln.encode_text(text)
                    repr = self.rln(input_dict)
                    output = fpln(repr)
                    loss = self.loss_fn(output, labels)
                    query_loss.append(loss.item())
                    with torch.no_grad():
                        if output.detach().size(1) == 1:
                            pred = (output > 0).int()
                        else:
                            pred = output.max(-1)[1]
                    acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())
                    query_acc.append(acc)
                    query_prec.append(prec)
                    query_rec.append(rec)
                    query_f1.append(f1)
                    it += 1

                    # RLN meta gradients
                    rln_params = [p for p in self.rln.parameters() if p.requires_grad]
                    meta_rln_grads = torch.autograd.grad(loss, rln_params, retain_graph=True)
                    for param, meta_grad in zip(rln_params, meta_rln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                    # PLN meta gradients
                    pln_params = [p for p in fpln.parameters() if p.requires_grad]
                    meta_pln_grads = torch.autograd.grad(loss, pln_params)
                    pln_params = [p for p in self.pln.parameters() if p.requires_grad]
                    for param, meta_grad in zip(pln_params, meta_pln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                # Meta optimizer step
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

                logger.info('Episode {} query set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
                                                                        np.mean(query_loss), np.mean(query_acc),
                                                                        np.mean(query_prec), np.mean(query_rec),
                                                                        np.mean(query_f1)))
                if self.online_logger:
                    self.online_logger.log_metrics({"loss": np.mean(query_loss), "accuracy":  np.mean(query_acc),
                                                    "precision": np.mean(query_prec), "recall": np.mean(query_rec),
                                                    "f1_score": np.mean(query_f1)}, prefix="train_query_",
                                                   step=episode_id+1)

                episode_id += 1

    def testing(self, test_datasets, **kwargs):
        updates = kwargs.get('updates')
        bs = kwargs.get('batch_size')
        accuracies, precisions, recalls, f1s = [], [], [], []
        for test_dataset in test_datasets:
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
            acc, prec, rec, f1 = self.evaluate(dataloader=test_dataloader, updates=updates,
                                               batch_size=bs)
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
        if self.online_logger:
            self.online_logger.log_metrics({"accuracy": np.mean(accuracies), "precision": np.mean(precisions),
                                            "recall": np.mean(recalls), "f1_score": np.mean(f1s)},
                                           prefix="test_ds_overall_")