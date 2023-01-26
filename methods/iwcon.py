import torch
from tqdm import tqdm
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

    def training_step(self, batch, batch_idx):
        iˢ, (s1, s2) = batch['source']
        iᵗ, (t1, t2) = batch['target']
        bˢ, bᵗ = len(iˢ), len(iᵗ)

        z = self.model(torch.cat((s1, t1)))
        z = self.head(z)
        zs1, zt1 = z.split([bˢ, bᵗ])

        with torch.no_grad():
            z = self.ema_model(torch.cat((s2, t2)))
            z = self.ema_head(z)
            zs2, zt2 = z.split([bˢ, bᵗ])

        self.update_queue('source', iˢ, zs2)
        self.update_queue('target', iᵗ, zt2)

        iw_lossˢ = self.iwcon(zs1, zs2, self.queue['source'])
        iw_lossᵗ = self.iwcon(zt1, zt2, self.queue['target'])
        iw_loss = iw_lossˢ + iw_lossᵗ

        return {
            'loss': iw_loss,
            'iwcon': {'loss_s': iw_lossˢ, 'loss_t': iw_lossᵗ}
        }

    def training_epoch_end(self, outputs):
        def agg(f):
            return torch.tensor(list(map(f, outputs))).mean()
        loss = agg(lambda x: x['loss'])
        iw_lossˢ = agg(lambda x: x['iwcon']['loss_s'])
        iw_lossᵗ = agg(lambda x: x['iwcon']['loss_t'])
        self.log_dict({
            'train/loss': loss,
            'train-iwcon/loss_s': iw_lossˢ,
            'train-iwcon/loss_t': iw_lossᵗ,
        }, sync_dist=self.is_distributed)
