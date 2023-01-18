from typing import Callable, Optional
from os.path import join
from operator import itemgetter
from torch.utils.data import Dataset
from bisect import bisect_left
from PIL import Image
from .base import BaseDataModule


def read_text(path):
    return map(str.split, open(path).readlines()[2:])


class DeepFashionDataset(Dataset):

    def __init__(self, root, domain, partition, transform=None):
        super().__init__()
        self.root = root
        self.domain = domain
        self.partition = partition
        self.transform = transform

        col = ['consumer', 'shop'].index(domain)

        data = read_text(join(root, 'Eval', 'list_eval_partition.txt'))
        data = [(x[col], x[2]) for x in data if x[3] == partition]
        data = sorted(set(data))
        items = sorted(set(map(itemgetter(1), data)))
        boxes = read_text(join(root, 'Anno', 'list_bbox_consumer2shop.txt'))
        boxes = {img: tuple(map(int, bbox)) for img, _, _,  *bbox in boxes}

        self.data = [(x, bisect_left(items, i), boxes[x]) for x, i in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, i, box = self.data[idx]
        x = join(self.root, 'Img', x)
        x = Image.open(x).convert('RGB').crop(box)
        if self.transform is not None:
            x = self.transform(x)
        if self.partition == 'train':
            return x
        return x, i


class DeepFashionDataModule(BaseDataModule):

    def get_raw_dataset(self, split: str, transform: Optional[Callable] = None):
        domain, partition = split.split('_')
        domain = {'source': 'consumer', 'target': 'shop'}[domain]
        return DeepFashionDataset(self.root, domain, partition, transform)
