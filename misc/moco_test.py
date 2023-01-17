import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torchvision import transforms as trfms
from methods.base import (
    load_moco_v2,
    pairwise_cosine_similarity,
    retrieval_mAP,
    retrieval_HR,
)
from datasets import DeepFashionDataModule
from tqdm import tqdm

ROOT = '/datasets/DeepFashion1/Consumer_to_Shop_Retrieval'

model = load_moco_v2()
model.eval()
model.to('cuda')

transform = trfms.Compose([
    trfms.Resize(256),
    trfms.CenterCrop(224),
    trfms.ToTensor(),
    trfms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


dm = DeepFashionDataModule(
    ROOT,
    transforms={'source_train': None, 'target_train': None, 'source_val': transform, 'target_val': transform},
    batch_sizes={'source_train': None, 'target_train': None, 'source_val': 32, 'target_val': 32},
)
dm.setup()
source_loader, target_loader = dm.test_dataloader()

with torch.no_grad():
    sources_v = []
    sources_c = []
    for x, c in tqdm(source_loader):
        v = model(x.cuda())
        sources_v.append(v)
        sources_c.append(c.cuda())

    targets_v = []
    targets_c = []
    for x, c in tqdm(target_loader):
        v = model(x.cuda())
        targets_v.append(v)
        targets_c.append(c.cuda())

sources_v = torch.concat(sources_v)
sources_c = torch.concat(sources_c)
targets_v = torch.concat(targets_v)
targets_c = torch.concat(targets_c)

simmat = pairwise_cosine_similarity(sources_v, targets_v)
preds = targets_c[simmat.argsort(descending=True)]
rels = (preds == sources_c.unsqueeze(-1))

mAP = retrieval_mAP(rels)
print('mAP:', mAP)

for k in [1, 5, 10, 20, 50]:
    v = retrieval_HR(rels, k)
    print(f'HR.{k}:', v)
