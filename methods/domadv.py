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

        z1, p1, _, p2 = self.forward_step(torch.cat(s1, t1), torch.cat(s2, t2))
        (ps1, pt1), (ps2, pt2) = p1.split([bˢ, bᵗ]), p2.split([bˢ, bᵗ])

        self.update_queue('source', iˢ, ps2)
        self.update_queue('target', iᵗ, pt2)

        iw_lossˢ = self.iwcon(ps1, ps2, self.queue['source'])
        iw_lossᵗ = self.iwcon(pt1, pt2, self.queue['target'])
        iw_loss = iw_lossˢ + iw_lossᵗ

        z_adv = self.domadv(z1)
        y_adv = torch.tensor([0] * bˢ + [1] * bᵗ).to(z_adv.device)
        adv_loss = self.domadv_loss(z_adv, y_adv)

        adv_λ = self.hparams.method['domadv_weight']

        loss = iw_loss + adv_λ * adv_loss

        return {
            'loss': loss,
            'iwcon': {'loss_s': iw_lossˢ, 'loss_t': iw_lossᵗ},
            'domadv': {'loss': adv_loss, 'weight': adv_λ},
        }

    def training_epoch_end(self, outputs):
        def agg(f):
            return torch.tensor(list(map(f, outputs))).mean()
        loss = agg(lambda x: x['loss'])
        iw_lossˢ = agg(lambda x: x['iwcon']['loss_s'])
        iw_lossᵗ = agg(lambda x: x['iwcon']['loss_t'])
        adv_loss = agg(lambda x: x['domadv']['loss'])
        self.log_dict({
            'train/loss': loss,
            'train-iwcon/loss_s': iw_lossˢ,
            'train-iwcon/loss_t': iw_lossᵗ,
            'train-domadv/loss': adv_loss,
        }, sync_dist=self.is_distributed)
