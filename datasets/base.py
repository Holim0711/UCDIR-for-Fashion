import os
from typing import Callable, Optional
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
        self.batch_sizes['source_test'] = self.batch_sizes['source_val']
        self.batch_sizes['target_test'] = self.batch_sizes['target_val']

    def get_raw_dataset(self, split: str, transform: Optional[Callable] = None):
        raise NotImplementedError()

    def setup(self, stage=None):
        self.datasets = {k: self.get_raw_dataset(k, t) for k, t in self.transforms.items()}
        self.datasets['source_test'] = self.get_raw_dataset('source_test', self.transforms['source_val'])
        self.datasets['target_test'] = self.get_raw_dataset('target_test', self.transforms['target_val'])

    def get_dataloader(self, split: str, **kwargs):
        dataset = self.datasets[split]
        batch_size = self.batch_sizes[split]
        kwargs.setdefault('num_workers', min(os.cpu_count() // 4, batch_size))
        kwargs.setdefault('pin_memory', True)
        if split.endswith('train'):
            dataset = IndexedDataset(dataset)
        return DataLoader(dataset, batch_size, **kwargs)

    def train_dataloader(self):
        return {'source': self.get_dataloader('source_train', shuffle=True),
                'target': self.get_dataloader('target_train', shuffle=True)}

    def val_dataloader(self):
        return [self.get_dataloader('source_val', shuffle=False),
                self.get_dataloader('target_val', shuffle=False)]

    def test_dataloader(self):
        return [self.get_dataloader('source_test', shuffle=False),
                self.get_dataloader('target_test', shuffle=False)]
