import os
from typing import Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from weaver.datasets import IndexedDataset


class BaseDataModule(LightningDataModule):

    def __init__(
        self,
        root: str,
        transforms: dict[str, Callable],
        batch_sizes: dict[str, int],
    ):
        super().__init__()
        self.root = root
        self.splits = ['source_train', 'target_train', 'source_val', 'target_val']
        self.transforms = {k: transforms[k] for k in self.splits}
        self.batch_sizes = {k: batch_sizes[k] for k in self.splits}
        for k in ['source', 'target']:
            self.transforms[k + '_test'] = self.transforms[k + '_val']
            self.batch_sizes[k + '_test'] = self.batch_sizes[k + '_val']

    def get_raw_dataset(self, split: str):
        raise NotImplementedError(f"self.get_raw_dataset('{split}')")

    def get_mapping_pairs(self, split: str):
        raise NotImplementedError(f"self.get_mapping_pairs('{split}')")

    def setup(self, stage=None):
        self.datasets = {k: self.get_raw_dataset(k) for k in self.splits}
        self.datasets['source_test'] = self.get_raw_dataset('source_test')
        self.datasets['target_test'] = self.get_raw_dataset('target_test')

    def get_dataloader(self, split: str, **kwargs):
        kwargs.setdefault('num_workers', min(os.cpu_count(), 32))
        kwargs.setdefault('pin_memory', True)
        dataset = self.datasets[split]
        if split.endswith('train'):
            dataset = IndexedDataset(dataset)
        return DataLoader(dataset, self.batch_sizes[split], **kwargs)

    def train_dataloader(self):
        return {'source': self.get_dataloader('source_train', shuffle=True),
                'target': self.get_dataloader('target_train', shuffle=True)}

    def val_dataloader(self):
        return [self.get_dataloader('source_val', shuffle=False),
                self.get_dataloader('target_val', shuffle=False)]
