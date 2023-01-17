import torch
from .base import BaseModule

__all__ = ['IWConModule']


class NTCrossEntropy(torch.nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, out_1, out_2):
        out = torch.cat([out_1, out_2], dim=0)
        n_samples = len(out)

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)

        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss


class IWConModule(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = NTCrossEntropy(self.hparams.method['temperature'])

    def all_gather_w_grad(self, x):
        X = self.all_gather(x)
        X[self.global_rank] = x
        return X.flatten(0, 1)

    def training_step(self, batch, batch_idx):
        iˢ, (q1, q2) = batch['source']
        iᵗ, (r1, r2) = batch['target']
        bˢ, bᵗ = len(iˢ), len(iᵗ)

        z = self.model(torch.cat((q1, q2, r1, r2)))
        z = self.head(z)
        zq1, zq2, zr1, zr2 = z.split([bˢ, bˢ, bᵗ, bᵗ])

        if self.trainer.world_size > 1:
            zq1 = self.all_gather_w_grad(zq1)
            zq2 = self.all_gather_w_grad(zq2)
            zr1 = self.all_gather_w_grad(zr1)
            zr2 = self.all_gather_w_grad(zr2)

        lossˢ = self.criterion(zq1, zq2) * self.trainer.world_size
        lossᵗ = self.criterion(zr1, zr2) * self.trainer.world_size
        loss = lossˢ + lossᵗ

        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bˢ + bᵗ)
        self.log('train/loss_q', lossˢ, on_step=False, on_epoch=True, sync_dist=True, batch_size=bˢ)
        self.log('train/loss_r', lossᵗ, on_step=False, on_epoch=True, sync_dist=True, batch_size=bᵗ)

        return {'loss': loss}
