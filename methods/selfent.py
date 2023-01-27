import torch
from .cwcon import CWConModule

__all__ = ['SelfEntropyModule']


class SelfEntropyLoss(torch.nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()
        self.τ = temperature

    def forward(self, features, centroids):
        z = torch.mm(features, centroids.t()) / self.τ
        z = z.softmax(dim=-1) * z.log_softmax(dim=-1)
        return -z.sum(dim=-1).mean()


class SelfEntropyModule(CWConModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.selfent = SelfEntropyLoss(self.hparams.method['selfent_temperature'])

    def training_step(self, batch, batch_idx):
        iˢ, (s1, s2) = batch['source']
        iᵗ, (t1, t2) = batch['target']
        bˢ, bᵗ = len(iˢ), len(iᵗ)
        cˢ, cᵗ = self.clusters['source'][iˢ], self.clusters['target'][iᵗ]

        x1, x2 = torch.cat([s1, t1]), torch.cat([s2, t2])
        _, p1, _, p2 = self.forward_step(x1, x2)
        (ps1, pt1), (ps2, pt2) = p1.split([bˢ, bᵗ]), p2.split([bˢ, bᵗ])

        self.update_queue('source', iˢ, ps2)
        self.update_queue('target', iᵗ, pt2)

        iw_lossˢ = self.iwcon(ps1, ps2, self.queue['source'])
        iw_lossᵗ = self.iwcon(pt1, pt2, self.queue['target'])
        iw_loss = iw_lossˢ + iw_lossᵗ

        cw_lossˢ = self.cwcon(ps1, cˢ, self.queue['source'], self.clusters['source'], self.centroids['source'])
        cw_lossᵗ = self.cwcon(pt1, cᵗ, self.queue['target'], self.clusters['target'], self.centroids['target'])
        cw_loss = cw_lossˢ + cw_lossᵗ

        se_lossˢ = self.selfent(ps1, self.centroids['source']) + self.selfent(ps1, self.centroids['target'])
        se_lossᵗ = self.selfent(pt1, self.centroids['target']) + self.selfent(ps1, self.centroids['source'])
        se_loss = se_lossˢ + se_lossᵗ

        t = self.current_epoch

        cw_λ = self.hparams.method['cwcon_weight']
        cw_start = self.hparams.method['cwcon_start']
        cw_warmup = self.hparams.method['cwcon_warmup']
        cw_λ *= min(1., max(0., (t - cw_start) / cw_warmup))

        se_λ = self.hparams.method['selfent_weight']
        se_λ *= (t >= self.hparams.method['selfent_start'])

        loss = iw_loss + cw_λ * cw_loss + se_λ * se_loss

        return {
            'loss': loss,
            'iwcon/loss_s': iw_lossˢ,
            'iwcon/loss_t': iw_lossᵗ,
            'cwcon/loss_s': cw_lossˢ,
            'cwcon/loss_t': cw_lossᵗ,
            'cwcon/weight': cw_λ,
            'selfent/loss_s': se_lossˢ,
            'selfent/loss_t': se_lossᵗ,
            'selfent/weight': se_λ,
        }

    def training_epoch_end(self, outputs):
        self.log_mean(outputs, [
            'iwcon/loss_s',
            'iwcon/loss_t',
            'cwcon/loss_s',
            'cwcon/loss_t',
            'cwcon/weight',
            'selfent/loss_s',
            'selfent/loss_t',
            'selfent/weight',
        ])
