import os
import torch
from pytorch_revgrad import RevGrad
from .iwcon import IWConModule

__all__ = ['DomAdvModule']


class DomAdvModule(IWConModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.domadv = torch.nn.Sequential(
            RevGrad(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2),
        )
        self.domadv_loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        iˢ, (s1, s2) = batch['source']
        iᵗ, (t1, t2) = batch['target']
        bˢ, bᵗ = len(iˢ), len(iᵗ)

        z = self.model(torch.cat((s1, t1)))

        z_adv = self.domadv(z)
        y_adv = torch.tensor([0] * bˢ + [1] * bᵗ).to(z_adv.device)
        adv_loss = self.domadv_loss(z_adv, y_adv)

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

        self.log('train-iwcon/loss', iw_loss, sync_dist=self.is_distributed)
        self.log('train-iwcon/loss_s', iw_lossˢ, sync_dist=self.is_distributed)
        self.log('train-iwcon/loss_t', iw_lossᵗ, sync_dist=self.is_distributed)
        self.log('train-adv/loss', adv_loss, sync_dist=self.is_distributed)

        return {'loss': iw_loss + self.hparams.method['domadv_weight'] * adv_loss}
