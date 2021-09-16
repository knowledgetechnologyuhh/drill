import logging
import torch
from torch import nn

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets
import models.utils
from models.base_models import BertBase, EpisodicMemory
from datasets.utils import ConcatDataset

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Replay-Log')


class Replay(object):

    def __init__(self, device, n_classes, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.write_probs = kwargs.get('write_probs')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.hidden_dim = kwargs.get('dim')
        self.device = device
        self.online_logger = kwargs.get('online_logger')
        
        self.model = BertBase(n_classes=n_classes, max_length=kwargs.get('max_len'), hidden_dim=self.hidden_dim,
                                  device=device)
        self.memory = EpisodicMemory(tuple_size=2, write_probs=self.write_probs)
        logger.info('Loaded {} as model'.format(self.model.__class__.__name__))

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def save_model(self, model_path):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

    def train(self, dataloader, dataset, epochs, log_freq):

        self.model.train()
        episode_id = 0

        for epoch in range(epochs):
            all_losses, all_predictions, all_labels = [], [], []
            itr = 0
            idx_pointer = 0
            for text, labels in dataloader:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = len(labels)
                replay_freq = self.replay_every // batch_size
                replay_steps = int(self.replay_every * self.replay_rate / batch_size)

                if self.replay_rate != 0 and (itr + 1) % replay_freq == 0:
                    self.optimizer.zero_grad()
                    for _ in range(replay_steps):
                        ref_text, ref_labels = self.memory.read_batch(batch_size=batch_size)
                        ref_labels = torch.tensor(ref_labels).to(self.device)
                        ref_input_dict = self.model.encode_text(ref_text)
                        ref_output = self.model(ref_input_dict)
                        ref_loss = self.loss_fn(ref_output, ref_labels)
                        ref_loss.backward()

                    params = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm(params, 25)
                    self.optimizer.step()

                loss = loss.item()
                pred = models.utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                episode_id += 1
                itr += 1
                idx_pointer += batch_size
                self.memory.write_batch(text, labels, ds_idx=dataset.get_idxs(idx_pointer)[0])

                if itr % log_freq == 0:
                    acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
                    logger.info(
                        'Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                        'F1 score = {:.4f}'.format(epoch + 1, np.mean(all_losses), acc, prec, rec, f1))

                    if self.online_logger:
                        self.online_logger.log_metrics({"loss": np.mean(all_losses), "accuracy": acc,
                                                        "precision": prec, "recall": rec,
                                                        "f1_score": f1}, prefix="train_query_",
                                                       step=episode_id)

                    all_losses, all_predictions, all_labels = [], [], []

    def evaluate(self, dataloader, test_id):
        all_losses, all_predictions, all_labels = [], [], []

        self.model.eval()

        for text, labels in dataloader:
            labels = torch.tensor(labels).to(self.device)
            input_dict = self.model.encode_text(text)
            with torch.no_grad():
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)
            loss = loss.item()
            pred = models.utils.make_prediction(output.detach())
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())

        acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))
        if self.online_logger:
            self.online_logger.log_metrics({"loss": np.mean(all_losses), "accuracy": acc,
                                            "precision": prec, "recall": rec, "f1_score": f1},
                                           prefix="test_ds_", step=test_id)

        return acc, prec, rec, f1

    def training(self, train_datasets, **kwargs):
        epochs = kwargs.get('epochs', 1)
        log_freq = kwargs.get('log_freq', 50)
        batch_size = kwargs.get('batch_size')
        train_dataset = ConcatDataset(train_datasets)
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                           collate_fn=datasets.utils.batch_encode)
        self.train(dataloader=train_dataloader, dataset=train_dataset, epochs=epochs, log_freq=log_freq)

    def testing(self, test_datasets, **kwargs):
        batch_size = kwargs.get('batch_size')
        accuracies, precisions, recalls, f1s = [], [], [], []
        for test_id, test_dataset in enumerate(test_datasets):
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=datasets.utils.batch_encode)
            acc, prec, rec, f1 = self.evaluate(dataloader=test_dataloader, test_id=test_id)
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
