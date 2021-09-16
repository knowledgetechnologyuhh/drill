import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from transformers import AdamW

import models.utils
from models.base_models import BertBase

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MTL-Log')


class Multitask:

    def __init__(self, device, n_classes, **kwargs):
        self.lr = kwargs.get('lr')
        self.hidden_dim = kwargs.get('dim')
        self.device = device
        self.model = BertBase(n_classes, kwargs.get('max_len'), self.hidden_dim, device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        self.online_logger = kwargs.get('online_logger')

    def training(self, train_datasets, **kwargs):
        log_freq = kwargs.get('log_freq')
        epochs = kwargs.get('epochs')
        episode_id = 0
        for epoch in range(epochs):
            logger.info('Epoch {}'.format(int(epoch)))
            train_dataset = data.ConcatDataset(train_datasets)
            logger.info('Training multi-task model on all datasets')
            train_dataloader = data.DataLoader(train_dataset, batch_size=kwargs.get('batch_size'), shuffle=True)
            self.model.train()

            all_losses, all_predictions, all_labels = [], [], []
            for it, (text, labels) in enumerate(train_dataloader):
                labels = torch.LongTensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                repr = self.model(input_dict, out_from='downsampler')
                output = self.model(repr, out_from='linear')
                loss = self.loss_fn(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                # make prediction
                with torch.no_grad():
                    if output.detach().size(1) == 1:
                        pred = (output > 0).int()
                    else:
                        pred = output.detach().max(-1)[1]
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                episode_id += 1

                if it % log_freq == 0:
                    acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
                    logger.info(
                        'Iteration {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                        'F1 score = {:.4f}'.format(it + 1, np.mean(all_losses), acc, prec, rec, f1))

                    if self.online_logger:
                        self.online_logger.log_metrics({"loss": np.mean(all_losses), "accuracy": acc,
                                                        "precision": prec, "recall": rec,
                                                        "f1_score": f1}, prefix="train_query_",
                                                       step=episode_id)

                    all_losses, all_predictions, all_labels = [], [], []

        # Returns the last training dataset since its the same regardless of the number of epochs
        return train_dataset

    def testing(self, test_datasets, **kwargs):
        accuracies, precisions, recalls, f1s = [], [], [], []

        for test_id, test_dataset in enumerate(test_datasets):
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=kwargs.get('batch_size'), shuffle=False)
            setattr(test_dataset, 'embeddings', pd.DataFrame(np.zeros((len(test_dataset), self.hidden_dim))))
            self.model.eval()

            all_losses, all_predictions, all_labels = [], [], []
            for it, (text, labels) in enumerate(test_dataloader):
                labels = torch.LongTensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                with torch.no_grad():
                    repr = self.model(input_dict, out_from='downsampler')
                    output = self.model(repr, out_from='linear')
                    loss = self.loss_fn(output, labels)
                loss = loss.item()

                # make prediction
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
            accuracies.append(acc), precisions.append(prec), recalls.append(rec), f1s.append(f1)
            if self.online_logger:
                self.online_logger.log_metrics({"loss": np.mean(all_losses), "accuracy": acc,
                                                "precision": prec, "recall": rec, "f1_score": f1s},
                                               prefix="test_ds_", step=test_id)

        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
        if self.online_logger:
            self.online_logger.log_metrics({"accuracy": np.mean(accuracies), "precision": np.mean(precisions),
                                            "recall": np.mean(recalls), "f1_score": np.mean(f1s)},
                                           prefix="test_ds_overall_")