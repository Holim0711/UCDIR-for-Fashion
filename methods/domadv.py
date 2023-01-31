import torch
from pytorch_revgrad import RevGrad
from .iwcon import IWConModule

__all__ = ['DomAdvModule']


class DomAdvModule(IWConModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.domadv = torch.nn.Sequential(
            RevGrad(self.hparams.method['domadv_alpha']),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2),
        )
        self.domadv_loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        iˢ, (s1, s2) = batch['source']
        iᵗ, (t1, t2) = batch['target']
        bˢ, bᵗ = len(iˢ), len(iᵗ)

        v1, z1 = self.model(torch.cat([s1, t1]))
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

        z_adv = self.domadv(z1)
        y_adv = torch.tensor([0] * bˢ + [1] * bᵗ).to(z_adv.device)
        adv_loss = self.domadv_loss(z_adv, y_adv)

        adv_λ = self.hparams.method['domadv_weight']

        loss = iw_loss + adv_λ * adv_loss

        return {
            'loss': loss,
            'iwcon/loss_s': iw_lossˢ,
            'iwcon/loss_t': iw_lossᵗ,
            'domadv/loss': adv_loss,
            'domadv/weight': adv_λ,
        }

    def training_epoch_end(self, outputs):
        self.log_mean(outputs, [
            'iwcon/loss_s',
            'iwcon/loss_t',
            'domadv/loss',
            'domadv/weight',
        ])
