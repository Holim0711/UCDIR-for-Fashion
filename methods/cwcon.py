import torch
import torch.distributed
# import faiss
from sklearn.cluster import KMeans
from .iwcon import IWConModule

__all__ = ['CWConModule']


def run_clustering(vectors: torch.Tensor, k):
    X = vectors.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    centroids = torch.tensor(kmeans.cluster_centers_)
    clusters = torch.tensor(kmeans.labels_, dtype=torch.int64)
    return centroids, clusters
    # vectors = vectors.cpu()
    # d = vectors.shape[1]
    # clus = faiss.Clustering(d, k)
    # clus.verbose = False
    # clus.niter = 20
    # clus.nredo = 5
    # clus.seed = 0
    # clus.max_points_per_centroid = 2000
    # clus.min_points_per_centroid = 2
    # cfg = faiss.GpuIndexFlatConfig()
    # cfg.useFloat16 = False
    # cfg.device = True
    # index = faiss.IndexFlatL2(d)
    # clus.train(vectors, index)
    # _, C = index.search(vectors, 1)
    # clusters = [int(x[0]) for x in C]
    # centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
    # return torch.tensor(centroids), torch.tensor(clusters)


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
        self.cwcon = CWConLoss(self.hparams.method['temperature'])
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

        cw_lossˢ = self.cwcon(zs1, cˢ, self.queue['source'], self.clusters['source'], self.centroids['source'])
        cw_lossᵗ = self.cwcon(zt1, cᵗ, self.queue['target'], self.clusters['target'], self.centroids['target'])
        cw_loss = cw_lossˢ + cw_lossᵗ

        λ = self.hparams.method['cwcon_weight']
        start = self.hparams.method['cwcon_start']
        warmup = self.hparams.method['cwcon_warmup']
        λ *= min(1., max(0., (self.current_epoch - start) / warmup))

        loss = iw_loss + λ * cw_loss

        return {
            'loss': loss,
            'iwcon': {'loss_s': iw_lossˢ, 'loss_t': iw_lossᵗ},
            'cwcon': {'loss_s': cw_lossˢ, 'loss_t': cw_lossᵗ,
                      'warm': λ, 'mask': self.cwcon.mask_ratio},
        }

    def training_epoch_end(self, outputs):
        def agg(f):
            return torch.tensor(list(map(f, outputs))).mean()
        loss = agg(lambda x: x['loss'])
        iw_lossˢ = agg(lambda x: x['iwcon']['loss_s'])
        iw_lossᵗ = agg(lambda x: x['iwcon']['loss_t'])
        cw_lossˢ = agg(lambda x: x['cwcon']['loss_s'])
        cw_lossᵗ = agg(lambda x: x['cwcon']['loss_t'])
        cw_warm = agg(lambda x: x['cwcon']['warm'])
        cw_mask = agg(lambda x: x['cwcon']['mask'])
        self.log_dict({
            'train/loss': loss,
            'train-iwcon/loss_s': iw_lossˢ,
            'train-iwcon/loss_t': iw_lossᵗ,
            'train-cwcon/loss_s': cw_lossˢ,
            'train-cwcon/loss_t': cw_lossᵗ,
            'train-cwcon/warm': cw_warm,
            'train-cwcon/mask': cw_mask,
        }, sync_dist=self.is_distributed)
