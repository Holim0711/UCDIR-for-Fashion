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
        super().__init__()
        self.iwcon = IWConLoss(self.hparams.method['temperature'])
        self.queue = {'sources': None, 'targets': None}

    def all_gather_w_grad(self, x):
        X = self.all_gather(x)
        X[self.global_rank] = x
        return X.flatten(0, 1)

    def prepare_deterministic_dataloaders(self, source_loader, target_loader):
        self.det_loaders = {'sources': source_loader, 'targets': target_loader}

    def on_train_start(self):
        self.ema_model.eval()
        self.ema_head.eval()
        with torch.no_grad():
            for k in self.queue:
                loader = tqdm(self.det_loaders[k], f'initialize queue for {k}')
                features = [self(x.to(self.device), head=True) for x in loader]
                self.queue[k] = torch.concat(features)

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

        if self.is_distributed:
            iˢ = self.all_gather(iˢ).flatten(0, 1)
            iᵗ = self.all_gather(iᵗ).flatten(0, 1)
            zs1 = self.all_gather_w_grad(zs1)
            zt1 = self.all_gather_w_grad(zt1)
            zs2 = self.all_gather(zs2).flatten(0, 1)
            zt2 = self.all_gather(zt2).flatten(0, 1)

        self.queue['sources'].index_copy_(0, iˢ, zs2)
        self.queue['targets'].index_copy_(0, iᵗ, zt2)

        iw_lossˢ = self.iwcon(zs1, zs2, self.queue['sources'])
        iw_lossᵗ = self.iwcon(zt1, zt2, self.queue['targets'])
        iw_loss = iw_lossˢ + iw_lossᵗ

        self.log('train-iwcon/loss', iw_loss, sync_dist=self.is_distributed)
        self.log('train-iwcon/loss_s', iw_lossˢ, sync_dist=self.is_distributed)
        self.log('train-iwcon/loss_t', iw_lossᵗ, sync_dist=self.is_distributed)

        return {'loss': iw_loss}
