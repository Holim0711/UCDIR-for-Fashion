import torch
import faiss
from tqdm import tqdm
from .iwcon import IWConModule

__all__ = ['CWConModule']


def run_clustering(vectors, k, gpu):
    vectors = vectors.cpu()
    d = vectors.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = False
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 2000
    clus.min_points_per_centroid = 2
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu
    index = faiss.IndexFlatL2(d)
    clus.train(vectors, index)
    _, C = index.search(vectors, 1)
    clusters = [int(x[0]) for x in C]
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
    return torch.tensor(centroids), torch.tensor(clusters)


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
        device = self.device
        nc = self.hparams.method['num_clusters']
        gpu = self.global_rank
        with torch.no_grad():
            for k in ['source', 'target']:
                loader = tqdm(self.det_loaders[k], f'run clustering for {k}')
                vectors = [self(x.to(device), with_head=True) for x in loader]
                vectors = torch.concat(vectors)
                centroids, clusters = run_clustering(vectors, nc, gpu)
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

        self.log('train-iwcon/loss', iw_loss, sync_dist=self.is_distributed)
        self.log('train-iwcon/loss_s', iw_lossˢ, sync_dist=self.is_distributed)
        self.log('train-iwcon/loss_t', iw_lossᵗ, sync_dist=self.is_distributed)
        self.log('train-cwcon/loss', cw_loss, sync_dist=self.is_distributed)
        self.log('train-cwcon/loss_s', cw_lossˢ, sync_dist=self.is_distributed)
        self.log('train-cwcon/loss_t', cw_lossᵗ, sync_dist=self.is_distributed)
        self.log('train-cwcon/mask', self.cwcon.mask_ratio, sync_dist=self.is_distributed)

        λ = self.hparams.method['cwcon_weight']
        start = self.hparams.method['cwcon_start']
        warmup = self.hparams.method['cwcon_warmup']
        λ *= min(1., max(0., (self.current_epoch - start) / warmup))

        return {'loss': iw_loss + λ * cw_loss}
