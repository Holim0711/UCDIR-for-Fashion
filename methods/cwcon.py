import torch
import torch.distributed
from sklearn.cluster import KMeans
from .iwcon import IWConModule

__all__ = ['CWConModule']


def run_clustering(vectors: torch.Tensor, k: int):
    X = vectors.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    centroids = torch.tensor(kmeans.cluster_centers_)
    clusters = torch.tensor(kmeans.labels_, dtype=torch.int64)
    return centroids, clusters


class CWConLoss(torch.nn.Module):

    def __init__(self, temperature=0.2, threshold=0.2):
        super().__init__()
        self.τ = temperature
        self.threshold = threshold

    def forward(self, q_v, q_c, k_v, k_c, centroids):
        # cluster-wise contrastive loss
        sim = torch.mm(q_v, k_v.t().contiguous()) / self.τ
        nll = sim.logsumexp(dim=-1, keepdim=True) - sim
        pos = (q_c.unsqueeze(-1) == k_c)
        loss = torch.sum(pos * nll, dim=-1) / pos.sum(dim=-1, keepdim=True)

        # filter out points far from centroids
        logits = torch.mm(q_v, centroids.t().contiguous()) / self.τ
        log_probs = logits.gather(1, q_c[:, None]) - logits.logsumexp(dim=-1)
        mask = (log_probs.exp() > self.threshold)
        loss = torch.sum(mask * loss) / mask.sum().clamp(min=1e-8)
        self.mask_ratio = mask.float().mean()
        return loss


class CWConModule(IWConModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.cwcon = CWConLoss(self.hparams.method['cwcon_temperature'])
        self.centroids = {'source': None, 'target': None}
        self.clusters = {'source': None, 'target': None}

    def on_train_epoch_start(self):
        nc = self.hparams.method['num_clusters']
        for k in ['source', 'target']:
            if not self.is_distributed or self.global_rank == 0:
                vectors = self.all_features(k, f'clustering for {k}')
                centroids, clusters = run_clustering(vectors, nc)
            else:
                centroids = torch.zeros(nc, 128)
                clusters = torch.zeros(len(self.queue[k]), dtype=int)
            if self.is_distributed:
                torch.distributed.broadcast(centroids, 0)
                torch.distributed.broadcast(clusters, 0)
            self.centroids[k] = centroids.to(self.device)
            self.clusters[k] = clusters.to(self.device)

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

        cw_λ = self.hparams.method['cwcon_weight']
        start = self.hparams.method['cwcon_start']
        warmup = self.hparams.method['cwcon_warmup']
        cw_λ *= min(1., max(0., (self.current_epoch - start) / warmup))

        loss = iw_loss + cw_λ * cw_loss

        return {
            'loss': loss,
            'iwcon/loss_s': iw_lossˢ,
            'iwcon/loss_t': iw_lossᵗ,
            'cwcon/loss_s': cw_lossˢ,
            'cwcon/loss_t': cw_lossᵗ,
            'cwcon/weight': cw_λ,
        }

    def training_epoch_end(self, outputs):
        self.log_mean(outputs, [
            'iwcon/loss_s',
            'iwcon/loss_t',
            'cwcon/loss_s',
            'cwcon/loss_t',
            'cwcon/weight',
        ])
