import torch
from tqdm import tqdm
from operator import itemgetter
from .base import BaseModule

__all__ = ['IWConModule']


class IWConLoss(torch.nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()
        self.τ = temperature

    def forward(self, query, pos, neg):
        neg = torch.mm(query, neg.t().contiguous()) / self.τ
        pos = torch.sum(query * pos, dim=-1) / self.τ
        return torch.mean(neg.logsumexp(dim=-1) - pos)


class IWConModule(BaseModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iwcon = IWConLoss(self.hparams.method['temperature'])
        self.queue = {'source': None, 'target': None}

    def prepare_deterministic_dataloaders(self, source_loader, target_loader):
        self.det_loaders = {'source': source_loader, 'target': target_loader}

    def all_features(self, domain, desc=None):
        d = self.device
        loader = tqdm(self.det_loaders[domain], desc)
        with torch.no_grad():
            features = [(self(x.to(d))) for x in loader]
        return torch.nn.functional.normalize(torch.concat(features))

    def on_train_start(self):
        super().on_train_start()
        for k in ['source', 'target']:
            self.queue[k] = self.all_features(k, f'initialize queue for {k}')

    def update_queue(self, domain, indices, vectors):
        if self.is_distributed:
            indices = self.all_gather(indices).flatten(0, 1)
            vectors = self.all_gather(vectors).flatten(0, 1)
        self.queue[domain][indices] = vectors

    def training_step(self, batch, batch_idx):
        iˢ, (s1, s2) = batch['source']
        iᵗ, (t1, t2) = batch['target']
        bˢ, bᵗ = len(iˢ), len(iᵗ)

        v1, _ = self.model(torch.cat([s1, t1]))
        v1 = torch.nn.functional.normalize(v1)

        with torch.no_grad():
            v2, _ = self.ema(torch.cat([s2, t2]))
            v2 = torch.nn.functional.normalize(v2)

        (vs1, vt1), (vs2, vt2) = v1.split([bˢ, bᵗ]), v2.split([bˢ, bᵗ])

        self.update_queue('source', iˢ, vs2)
        self.update_queue('target', iᵗ, vt2)

        iw_lossˢ = self.iwcon(vs1, vs2, self.queue['source'])
        iw_lossᵗ = self.iwcon(vt1, vt2, self.queue['target'])
        iw_loss = iw_lossˢ + iw_lossᵗ

        return {
            'loss': iw_loss,
            'iwcon/loss_s': iw_lossˢ,
            'iwcon/loss_t': iw_lossᵗ,
        }

    def log_mean(self, outputs, keys):
        def aggr(k):
            return torch.tensor(list(map(itemgetter(k), outputs))).mean()
        self.log_dict({
            'train/loss': aggr('loss'),
            **{'train-' + k: aggr(k) for k in keys}
        }, sync_dist=self.is_distributed)

    def training_epoch_end(self, outputs):
        self.log_mean(outputs, [
            'iwcon/loss_s',
            'iwcon/loss_t',
        ])
