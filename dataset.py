import functools
import os
from typing import Callable, Optional, Sequence, Union

import gin
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import tqdm

# Disable tqdm globally.
# tqdm.tqdm.__init__ = functools.partialmethod(tqdm.tqdm.__init__, disable=True)


@gin.configurable
class Bvcc(Dataset):
    """The BVCC dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/BVCC/DATA',
        features_folder: str = 'w2v2_xlsr_2b_10s/layer7',
        valid: str = 'train',
        debug: bool = False,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
        """
        self._data_path = data_path
        self._features_folder = features_folder
        self._debug = debug
        self._df = self._load_df(valid)
        self._num_samples = len(self._df)
        self._valid = valid

        self._systems, self._features, self._labels = self._load_clips()

    @property
    def features_shape(self) -> int:
        return self._features[0].shape

    def _load_df(self, valid: str) -> pd.DataFrame:
        if valid not in ['train', 'val', 'test']:
            raise ValueError(f'{valid=} is not valid.')
        # read metadata
        systems, filenames, labels = [], [], []
        with open(f'../../datasets/BVCC/DATA/sets/{valid}_mos_list.txt', 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if line:
                    filename, label = line.split(",")
                    systems.append(filename[:8])
                    filenames.append(filename)
                    labels.append(float(label))
                    if self._debug and len(filenames) == 100:
                        break

        df = pd.DataFrame({'systems': systems, 'filenames': filenames, 'labels': labels})
        return df

    @property
    def unique_systems(self) -> set[str]:
        return set(self._df['systems'])

    def _load_clips(self):
        """Loads the clips, applies augmentations (if so), and transforms to spectrograms"""
        systems, features, labels = [], [], []
        for system, filename, label in tqdm.tqdm(zip(self._df['systems'], self._df['filenames'], self._df['labels']), total=self._num_samples, desc='Loading clips...'):
            try:
                feature = np.load(os.path.join(self._data_path, self._features_folder, filename.replace('.wav', '.npy')))
                systems.append(system)
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')

        return systems, features, labels

    def __getitem__(self, idx: int):
        """Returns a spectrogram with label and augmentation applied."""
        return self._systems[idx], self._features[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._features)

    def collate_fn(self, batch: list):
        """Returns a batch consisting of tensors."""
        systems, features, labels = zip(*batch)
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(labels)
        return systems, features, labels


@gin.configurable
class Tencent(Dataset):
    """The Tencent corpus dataset."""

    def __init__(
        self,
        data_path: str = '../../datasets/Tencent_corpus',
        features_folder: str = 'w2v2_base/layer7',
        valid: str = 'train',
        debug: bool = False,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
        """
        self._data_path = data_path
        self._features_folder = features_folder
        self._debug = debug

        self._df = self._load_df(valid)
        self._num_samples = len(self._df)
        self._valid = valid

        self._systems, self._features, self._labels = self._load_clips()
    
    @property
    def features_shape(self) -> int:
        return self._features[0].shape

    def _load_df(self, valid: str) -> pd.DataFrame:
        without_reverb_df = pd.read_csv(os.path.join(self._data_path, 'withoutReverberationTrainDevMOS.csv'))
        with_reverb_df = pd.read_csv(os.path.join(self._data_path, 'withReverberationTrainDevMOS.csv'))
        df = pd.concat([without_reverb_df, with_reverb_df]).sample(frac=1, random_state=1997)
        if valid == 'all':
            pass
        elif valid == 'train':
            df = df.iloc[:8000]
        elif valid == 'val':
            df = df.iloc[8000:10000]
        elif valid == 'test':
            df = df.iloc[10000:]
        else:
            raise ValueError(f'{valid=} is not valid.')

        df = df.rename(columns={'deg_wav': 'filenames', 'mos': 'labels'})
        return df
    
    @property
    def unique_systems(self) -> set[str]:
        return set(self._df['systems'])

    def _load_clips(self):
        """Loads the clips, applies augmentations (if so), and transforms to spectrograms"""
        systems, features, labels = [], [], []
        for filename, label in tqdm.tqdm(zip(self._df['filenames'], self._df['labels']), total=self._num_samples, desc='Loading clips...'):
            try:
                _, subfolder, subfile = filename.split('/')
                feature = np.load(os.path.join(self._data_path, subfolder+'_features', self._features_folder, subfile.replace('.wav', '.npy')))
                systems.append('system')
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')

        return systems, features, labels

    def __getitem__(self, idx: int):
        """Returns a spectrogram with label and augmentation applied."""
        return self._systems[idx], self._features[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._features)

    def collate_fn(self, batch: list):
        """Returns a batch consisting of tensors."""
        systems, features, labels = zip(*batch)
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(labels)
        return systems, features, labels


@gin.configurable
class Nisqa(Dataset):
    """The NISQA dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/NISQA_Corpus',
        features_folder: str = 'w2v2_base/layer7',
        valid: str = 'train',
        debug: bool = False,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
        """
        self._data_path = data_path
        self._features_folder = features_folder
        self._debug = debug

        self._df = self._load_df(valid)
        self._num_samples = len(self._df)
        self._valid = valid

        self._systems, self._features, self._labels = self._load_clips()
    
    @property
    def features_shape(self) -> int:
        return self._features[0].shape

    def _load_df(self, valid: str) -> pd.DataFrame:
        if valid == 'train':
            train_sim_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TRAIN_SIM', 'NISQA_TRAIN_SIM_file.csv'))
            train_live_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TRAIN_LIVE', 'NISQA_TRAIN_LIVE_file.csv'))
            df = pd.concat([train_sim_df, train_live_df])
        elif valid == 'val':
            val_sim_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_VAL_SIM', 'NISQA_VAL_SIM_file.csv'))
            val_live_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_VAL_LIVE', 'NISQA_VAL_LIVE_file.csv'))
            df = pd.concat([val_sim_df, val_live_df])
        elif valid == 'test':
            test_for_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TEST_LIVETALK', 'NISQA_TEST_LIVETALK_file.csv'))
            df = test_for_df
        else:
            raise ValueError(f'{valid=} is not valid.')
        df = df[['db', 'filename_deg','mos']]
        df = df.rename(columns={'db': 'foldernames','filename_deg': 'filenames', 'mos': 'labels'})
        return df

    @property
    def unique_systems(self) -> set[str]:
        return set(self._df['systems'])

    def _load_clips(self):
        """Loads the clips, applies augmentations (if so), and transforms to spectrograms"""
        systems, features, labels = [], [], []
        for splitfoldername, filename, label in tqdm.tqdm(zip(self._df['foldernames'], self._df['filenames'], self._df['labels']), total=self._num_samples, desc='Loading clips...'):
            try:
                feature = np.load(os.path.join(self._data_path, 'features', splitfoldername, self._features_folder, filename.replace('.wav', '.npy')))
                systems.append('system')
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')

        return systems, features, labels

    def __getitem__(self, idx: int):
        """Returns a spectrogram with label and augmentation applied."""
        return self._systems[idx], self._features[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._features)

    def collate_fn(self, batch: list):
        """Returns a batch consisting of tensors."""
        systems, features, labels = zip(*batch)
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(labels)
        return systems, features, labels


@gin.configurable
def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Returns a dataloader of the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
