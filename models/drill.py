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
from models.base_models import BertRLN, EpisodicMemory, GatingPLN
from datasets.utils import ConcatDataset

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DRILL-Log')


class DRILL:
    def __init__(self, device, n_classes, gm, fusion="gate", **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_probs = kwargs.get('write_probs')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.hidden_dim = kwargs.get('dim')
        self.supervised_sampling = kwargs.get('supervised_sampling', False)
        self.device = device
        self.online_logger = kwargs.get('online_logger')

        self.gm = gm.to(device)
        self.rln = BertRLN(kwargs.get('max_len'), self.hidden_dim, device, activate=False)
        self.pln = GatingPLN(in_dim=self.hidden_dim, out_dim=n_classes, fusion=fusion, device=device)
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
        self.gm.train()
        self.pln.train()

        support_set = []
        for _ in range(updates):
            text, labels = self.memory.read_batch(batch_size=batch_size)
            support_set.append((text, labels))

        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpln, diffopt):

            # Inner loop
            task_predictions, mem_task_predictions, task_labels = [], [], []
            support_loss = []
            for it, (text, labels) in enumerate(support_set):
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.rln.encode_text(text)
                repr = self.rln(input_dict)
                gm_pred, bmus = self.gm(it, zip(repr, labels), n_bmus=1, supervised_sampling=self.supervised_sampling)
                output = fpln(repr, bmus.squeeze(1))
                loss = self.loss_fn(output, labels)
                diffopt.step(loss)
                with torch.no_grad():
                    if output.detach().size(1) == 1:
                        pred = (output > 0).int()
                    else:
                        pred = output.max(-1)[1]
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                mem_task_predictions.extend(gm_pred)
                task_labels.extend(labels.tolist())
            acc, prec, rec, f1 = {}, {}, {}, {}
            acc['model'], prec['model'], rec['model'], f1['model'] = models.utils.calculate_metrics(task_predictions,
                                                                                                    task_labels)
            acc['memory'], prec['memory'], rec['memory'], f1['memory'] = models.utils.calculate_metrics(
                mem_task_predictions, task_labels)

            logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                        'recall = {:.4f}, F1 score = {:.4f}'.format(
                np.mean(support_loss), acc['model'], prec['model'], rec['model'], f1['model']))
            if self.online_logger:
                self.online_logger.log_metrics({"loss": np.mean(support_loss),
                                                "accuracy": acc['model'], "precision": prec['model'],
                                                "recall": rec['model'], "f1_score": f1['model']}, prefix="eval_supp_")

                self.online_logger.log_metrics({"accuracy": acc['memory'], "precision": prec['memory'],
                                                "recall": rec['memory'], "f1_score": f1['memory']},
                                               prefix="eval_supp_mem_")

            all_losses, all_predictions, all_mem_predictions, all_labels = [], [], [], []
            self.gm.eval()
            for it, (text, labels) in enumerate(dataloader):
                labels = torch.LongTensor(labels).to(self.device)
                input_dict = self.rln.encode_text(text)
                with torch.no_grad():
                    repr = self.rln(input_dict)
                    gm_pred, bmus = self.gm(it, zip(repr, labels), n_bmus=1,
                                            supervised_sampling=self.supervised_sampling)
                    output = fpln(repr, bmus.squeeze(1))
                    loss = self.loss_fn(output, labels)
                loss = loss.item()
                with torch.no_grad():
                    if output.detach().size(1) == 1:
                        pred = (output > 0).int()
                    else:
                        pred = output.max(-1)[1]
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_mem_predictions.extend(gm_pred)
                all_labels.extend(labels.tolist())

        acc, prec, rec, f1 = {}, {}, {}, {}
        acc['model'], prec['model'], rec['model'], f1['model'] = models.utils.calculate_metrics(all_predictions,
                                                                                                all_labels)
        acc['memory'], prec['memory'], rec['memory'], f1['memory'] = models.utils.calculate_metrics(all_mem_predictions,
                                                                                                    all_labels)

        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(all_losses), acc['model'], prec['model'], rec['model'],
                                               f1['model']))
        if self.online_logger:
            self.online_logger.log_metrics(
                {"loss": np.mean(all_losses), "accuracy": acc['model'], "precision": prec['model'],
                 "recall": rec['model'], "f1_score": f1['model']}, prefix="eval_")
            self.online_logger.log_metrics(
                {"accuracy": acc['memory'], "precision": prec['memory'],
                 "recall": rec['memory'], "f1_score": f1['memory']}, prefix="eval_mem_")
        return acc, prec, rec, f1

    def training(self, train_datasets, **kwargs):
        updates = kwargs.get('updates')
        batch_size = kwargs.get('batch_size')
        idx_pointer = 0

        memory_iter = kwargs.get('batch_size', batch_size)

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
                    gm_pred, bmus = self.gm(it, zip(repr, labels), n_bmus=1,
                                            supervised_sampling=self.supervised_sampling)
                    # fusion, state = self.gbu((repr, torch.zeros_like(repr, device=self.device)))
                    output = fpln(repr, bmus.squeeze(1))
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
                                                   step=episode_id + 1)
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
                    gm_pred, bmus = self.gm(it, zip(repr, labels), n_bmus=memory_iter,
                                            supervised_sampling=self.supervised_sampling)

                    # repeat model for SOINN neighbors
                    for repeat in range(memory_iter):
                        labels_replica = labels.clone()
                        input_dict_replica = self.rln.encode_text(text)
                        repr_replica = self.rln(input_dict_replica)
                        output = fpln(repr_replica, bmus[:, repeat])
                        loss = self.loss_fn(output, labels_replica)
                        query_loss.append(loss.item())
                        with torch.no_grad():
                            if output.detach().size(1) == 1:
                                pred = (output > 0).int()
                            else:
                                pred = output.max(-1)[1]
                        acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels_replica.tolist())
                        query_acc.append(acc)
                        query_prec.append(prec)
                        query_rec.append(rec)
                        query_f1.append(f1)

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
                    it += 1

                # Outer-loop meta optimization step
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

                logger.info('Episode {} query set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
                                                                        np.mean(query_loss), np.mean(query_acc),
                                                                        np.mean(query_prec), np.mean(query_rec),
                                                                        np.mean(query_f1)))
                if self.online_logger:
                    self.online_logger.log_metrics({"loss": np.mean(query_loss), "accuracy": np.mean(query_acc),
                                                    "precision": np.mean(query_prec), "recall": np.mean(query_rec),
                                                    "f1_score": np.mean(query_f1)}, prefix="train_query_",
                                                   step=episode_id + 1)
                episode_id += 1

    def testing(self, test_datasets, **kwargs):
        updates = kwargs.get('updates')
        bs = kwargs.get('batch_size')

        accuracies, precisions, recalls, f1s = [], [], [], []
        mem_accuracies, mem_precisions, mem_recalls, mem_f1s = [], [], [], []

        for test_dataset in test_datasets:
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
            acc, prec, rec, f1 = self.evaluate(dataloader=test_dataloader, updates=updates,
                                               batch_size=bs)
            accuracies.append(acc['model'])
            precisions.append(prec['model'])
            recalls.append(rec['model'])
            f1s.append(f1['model'])

            mem_accuracies.append(acc['memory'])
            mem_precisions.append(prec['memory'])
            mem_recalls.append(rec['memory'])
            mem_f1s.append(f1['memory'])

        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
        if self.online_logger:
            self.online_logger.log_metrics({"accuracy": np.mean(accuracies), "precision": np.mean(precisions),
                                            "recall": np.mean(recalls), "f1_score": np.mean(f1s)},
                                           prefix="test_ds_overall_")

            self.online_logger.log_metrics({"accuracy": np.mean(mem_accuracies), "precision": np.mean(mem_precisions),
                                            "recall": np.mean(mem_recalls), "f1_score": np.mean(mem_f1s)},
                                           prefix="test_ds_mem_overall_")
