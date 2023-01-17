import os
from typing import Callable
from itertools import chain
from torch.utils.data import Dataset
from PIL import Image
from .base import BaseDataModule


class OfficeHomeDataset(Dataset):

    def __init__(self, root, domain, partition, classes=None, transform=None):
        super().__init__()
        self.root = root
        self.domain = domain
        self.partition = partition
        self.classes = classes
        self.transform = transform

        if classes is None:
            self.classes = sorted(os.listdir(os.path.join(root, domain)))

        dirs = [os.path.join(root, domain, c) for c in self.classes]
        data = [[(x, c) for x in os.listdir(d)] for c, d in enumerate(dirs)]
        data = sorted(chain(*data))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, c = self.data[idx]
        x = os.path.join(self.root, self.domain, self.classes[c], x)
        x = Image.open(x).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        if self.partition == 'train':
            return x
        return x, c


class OfficeHomeDataModule(BaseDataModule):

    def __init__(
        self,
        root: str,
        source_domain: str,
        target_domain: str,
        transforms: dict[str, Callable],
        batch_sizes: dict[str, int],
    ):
        super().__init__(root, transforms, batch_sizes)
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.classes = sorted(os.listdir(os.path.join(root, source_domain)))

    def get_raw_dataset(self, split: str):
        transform = self.transforms.get(split)
        dom, part = split.split('_')
        dom = {'source': self.source_domain, 'target': self.target_domain}[dom]
        return OfficeHomeDataset(self.root, dom, part, self.classes, transform)


class DomainNetDataModule(OfficeHomeDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = [
            'bird', 'feather', 'teapot', 'tiger', 'whale', 'windmill', 'zebra',
        ]
