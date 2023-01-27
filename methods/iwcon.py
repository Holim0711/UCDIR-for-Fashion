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
            features = [self.ema_head(self.ema_model(x.to(d))) for x in loader]
        return torch.concat(features)

    def on_train_start(self):
        super().on_train_start()
        for k in ['source', 'target']:
            self.queue[k] = self.all_features(k, f'initialize queue for {k}')

    def update_queue(self, domain, indices, vectors):
        if self.is_distributed:
            indices = self.all_gather(indices).flatten(0, 1)
            vectors = self.all_gather(vectors).flatten(0, 1)
        self.queue[domain][indices] = vectors

    def forward_step(self, x1, x2):
        z1 = self.model(x1)
        p1 = self.head(z1)
        with torch.no_grad():
            z2 = self.ema_model(x2)
            p2 = self.ema_head(z2)
        return z1, p1, z2, p2

    def training_step(self, batch, batch_idx):
        iˢ, (s1, s2) = batch['source']
        iᵗ, (t1, t2) = batch['target']
        bˢ, bᵗ = len(iˢ), len(iᵗ)

        x1, x2 = torch.cat([s1, t1]), torch.cat([s2, t2])
        _, p1, _, p2 = self.forward_step(x1, x2)
        (ps1, pt1), (ps2, pt2) = p1.split([bˢ, bᵗ]), p2.split([bˢ, bᵗ])

        self.update_queue('source', iˢ, ps2)
        self.update_queue('target', iᵗ, pt2)

        iw_lossˢ = self.iwcon(ps1, ps2, self.queue['source'])
        iw_lossᵗ = self.iwcon(pt1, pt2, self.queue['target'])
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
