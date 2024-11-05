import argparse
import copy
import logging
import os
import shutil
from typing import Any, Optional, Type

import gin
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import tqdm

from dataset import Bvcc, get_dataloader
from model import ProjectionHead


parser = argparse.ArgumentParser(description='Gin and save path.')
parser.add_argument('--gin_path', type=str, help='Path to the gin-config.')
parser.add_argument('--save_path', type=str, help='Path to directory storing results.')
parser.add_argument('--features_folder', type=str, help='Folder of the features.')

args = parser.parse_args()


@gin.configurable
class TrainingLoop:
    """The training loop which trains and evaluates a model."""

    def __init__(
        self,
        *,
        model: nn.Module = ProjectionHead,
        save_path: str = '',
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        weight_decay: float = 0.0,
        dataset_cls: Type[Dataset] = Bvcc,
        num_epochs: int = 500,
        learning_rate: float = 1e-4,
        batch_size_train: int = 64,
        loss_type: str = 'mse',
    ):
        """Initializes the instance.
        
        Args:
            model: The nn.Module model. Expected to take (B, 1, T, F) as input,
                and output (B, S), where B is the batch size, T is the time bins,
                F is the frequency bins, and S is the score.
            save_path: Path to log directory.
            loss_type: Type of loss, 'mse' or 'mae' are supported.
            optimizer: The optimizer.
            weight_decay: Weight decay of the parameters.
            dataset_cls: The dataset class. Expected to have the parameter
                `valid`, which can take the values 'train', 'val', and 'test'.
                Returns a `Dataset` object.
            num_epochs: Number of training epochs.
            learning_rate: The learning rate.
            batch_size_train: Batch size of train set.
        """
        # Setup logging and paths.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print('New directory added!')
        log_path = os.path.join(save_path, 'train.log')
        self._save_path = save_path
        logging.basicConfig(filename=log_path, level=logging.INFO)

        # Datasets.
        train_dataset = dataset_cls(valid='train', features_folder=args.features_folder)
        valid_dataset = dataset_cls(valid='val', features_folder=args.features_folder)
        test_dataset = dataset_cls(valid='test', features_folder=args.features_folder)
        logging.info(f'Num train speech clips: {len(train_dataset)}')
        logging.info(f'Num val speech clips: {len(valid_dataset)}')
        logging.info(f'Num test speech clips: {len(test_dataset)}')
        self._train_loader = get_dataloader(dataset=train_dataset, batch_size=batch_size_train)
        self._valid_loader = get_dataloader(dataset=valid_dataset, batch_size=1)
        self._test_loader = get_dataloader(dataset=test_dataset, batch_size=1)

        # Model and optimizers.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'Device={self._device}')
        self._model = model(in_shape=train_dataset.features_shape).to(self._device)

        self._best_pcc = -1
        # TODO: Explore some learning rate scheduler.
        self._optimizer = optimizer(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._optimizer.zero_grad()
        self._lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self._optimizer, gamma=0.9999)
        self._loss_type = loss_type
        if loss_type == 'mse':
            self._loss_fn = F.mse_loss
        elif loss_type == 'mae':
            self._loss_fn = F.l1_loss
        elif loss_type == 'gnll':
            self._loss_fn = F.gaussian_nll_loss
        else:
            raise ValueError(f'Loss {loss_type} not supported.')

        self._all_loss = []
        self._epoch = 0
        self._num_epochs = num_epochs
        self._save_every_n_epochs = 10

    @property
    def save_path(self):
        """The path to the log directory."""
        return self._save_path
    
    def train(self, valid_each_epoch: bool = True) -> None:
        """Trains the model on the train data `self._num_epochs` number of epochs.
        
        Args:
            valid_each_epoch: If to compute the validation performance.
        """
        self._model.train()
        while self._epoch <= self._num_epochs:
            self._all_loss = list()
            for batch in tqdm.tqdm(self._train_loader, ncols=0, desc="Train", unit=" step"):
                _, features, labels = batch
                features = features.to(self._device)
                labels = labels.to(self._device)

                # Forward
                predictions = self._model(features).to(self._device).squeeze()
                predictions = predictions.squeeze(-1)
                loss = self._loss_fn(labels, predictions)
                
                # Backwards
                loss.backward()
                self._all_loss.append(loss.item())
                del loss

                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)

                # Step
                self._optimizer.step()
                self._lr_scheduler.step()
                self._optimizer.zero_grad()

            average_loss = np.mean(self._all_loss)
            logging.info(f'Average loss={average_loss}')

            if valid_each_epoch:
                self._evaluate(self._valid_loader, 'Valid')

            self._epoch += 1
        

    def _evaluate(self, dataloader: Any, prefix: str):
        """Evaluates the model on the data based on quality prediction and augmentation accuracy."""
        self._model.eval()
        predictions, labels = [], []
        # system_predictions = {system: [] for system in dataloader.dataset.unique_systems}
        # system_labels = copy.deepcopy(system_predictions)
        for i, batch in enumerate(tqdm.tqdm(dataloader, ncols=0, desc=prefix, unit=' step')):
            system, feature, mos = batch
            # system_name = system[0]
            feature = feature.to(self._device)

            with torch.no_grad():
                try:
                    prediction = self._model(feature)
                    prediction = prediction.squeeze(-1)

                    predictions.extend(prediction.cpu().tolist())
                    labels.extend(mos.tolist())
                    # system_predictions[system_name].extend(prediction.cpu().tolist())
                    # system_labels[system_name].extend(mos.tolist())
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        logging.error(f'[Runner] - CUDA out of memory.')
                        with torch.cuda.device(self._device):
                            torch.cuda.empty_cache()
                    else:
                        raise

        predictions = np.array(predictions)
        labels = np.array(labels)
        utt_mse = np.mean((labels-predictions)**2)
        utt_pcc = np.corrcoef(labels, predictions)[0][1]
        utt_srcc = scipy.stats.spearmanr(labels, predictions)[0]
        if utt_pcc > self._best_pcc:
            self._best_pcc = utt_pcc
            self.save_model('model_best.pt')
        # system_predictions_array = np.array([np.mean(scores) for scores in system_predictions.values()])
        # system_labels_array = np.array([np.mean(scores) for scores in system_labels.values()])
        # sys_mse=np.mean((system_labels_array-system_predictions_array)**2)
        # sys_pcc=np.corrcoef(system_labels_array, system_predictions_array)[0][1]
        # sys_srcc=scipy.stats.spearmanr(system_labels_array, system_predictions_array)[0]

        logging.info(f"\n[{prefix}][{self._epoch}][UTT][ MSE = {utt_mse:.4f} | LCC = {utt_pcc:.4f} | SRCC = {utt_srcc:.4f} ]")
        # logging.info(f"\n[{prefix}][{self._epoch}][SYS][ MSE = {sys_mse:.4f} | LCC = {sys_pcc:.4f} | SRCC = {sys_srcc:.4f} ]")
        self._model.train()
    
    def test(self):
        """Evaluates the model on test data."""
        self._model = torch.jit.load(os.path.join(self._save_path, 'model_best.pt')).to(self._device)
        self._evaluate(self._test_loader, 'Test')

    def save_model(self, model_name: str = 'model.pt'):
        """Saves the model."""
        model_scripted = torch.jit.script(self._model)
        model_scripted.save(os.path.join(self._save_path, model_name))


def main():
    """Main."""
    gin.external_configurable(torch.nn.modules.activation.ReLU, module='torch.nn.modules.activation')
    gin.external_configurable( torch.nn.modules.activation.SiLU, module='torch.nn.modules.activation')
    gin.parse_config_file(args.gin_path)

    train_loop = TrainingLoop(save_path=args.save_path)
    new_gin_path = os.path.join(train_loop.save_path, 'config.gin')
    shutil.copyfile(args.gin_path, new_gin_path)
    train_loop.train()
    train_loop.test()
    train_loop.save_model()


if __name__ == '__main__':
    main()
