import torch
import pytorch_lightning as pl
from weaver import get_optimizer, get_scheduler
from .utils import retrieval_report, load_moco_v2, BNN, CGD


def change_momentum(model: torch.nn.Module, momentum: float):
    if isinstance(model, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
        model.momentum = 1 - momentum
    for child in model.children():
        change_momentum(child, momentum)


class EMA(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model: torch.nn.Module, a: float):
        super().__init__(model, avg_fn=lambda m, x, _: a * m + (1 - a) * x)

    def update_parameters(self, model):
        super().update_parameters(model)
        # BatchNorm buffers are already EMA
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


def pairwise_cosine_similarity(x1, x2):
    x1 = torch.nn.functional.normalize(x1)
    x2 = torch.nn.functional.normalize(x2)
    return torch.mm(x1, x2.t())


def result_check(vˢ, vᵗ, cˢ, cᵗ):
    sim = pairwise_cosine_similarity(vˢ, vᵗ).cpu()
    rel = cᵗ[sim.argsort(descending=True)] == cˢ.unsqueeze(-1)
    return rel.cpu()


class BaseModule(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = load_moco_v2(remove_last_downsampling=False)
        # self.model.head = BNN()
        # self.model.head = CGD()

        change_momentum(self.model, self.hparams.ema)
        self.ema = EMA(self.model, self.hparams.ema)
        self.ema.eval()

    def forward(self, x):
        return self.ema(x)[0]

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update_parameters(self.model)

    def on_train_start(self):
        super().on_train_start()
        self.ema.eval()

    def on_validation_model_train(self):
        super().on_validation_model_train()
        self.ema.eval()

    def setup(self, stage=None):
        self.is_distributed = torch.distributed.is_initialized()

    def shared_step(self, batch, batch_idx, dataloader_idx):
        x, c = batch
        v, z = self.ema(x)
        return {'z': z.half(), 'v': v.half(), 'c': c}

    def shared_epoch_end(self, stage, outputs):
        sources, targets = outputs
        vˢ = torch.concat([x['v'] for x in sources])
        zˢ = torch.concat([x['z'] for x in sources])
        cˢ = torch.concat([x['c'] for x in sources])
        vᵗ = torch.concat([x['v'] for x in targets])
        zᵗ = torch.concat([x['z'] for x in targets])
        cᵗ = torch.concat([x['c'] for x in targets])

        if self.is_distributed:
            zᵗ = self.all_gather(zᵗ).flatten(0, 1)
            vᵗ = self.all_gather(vᵗ).flatten(0, 1)
            cᵗ = self.all_gather(cᵗ).flatten(0, 1)

        rels = result_check(vˢ, vᵗ, cˢ, cᵗ)
        report = retrieval_report(rels, self.hparams.dataset['metric'])
        self.log_dict({f'p-{stage}/{k}': v for k, v in report.items()},
                      sync_dist=self.is_distributed)

        z_rels = result_check(zˢ, zᵗ, cˢ, cᵗ)
        z_report = retrieval_report(z_rels, self.hparams.dataset['metric'])
        self.log_dict({f'z-{stage}/{k}': v for k, v in z_report.items()},
                      sync_dist=self.is_distributed)

    def validation_step(self, *args, **kwargs):
        return self.shared_step(*args, **kwargs)

    def validation_epoch_end(self, *args, **kwargs):
        return self.shared_epoch_end('val', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.shared_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        return self.shared_epoch_end('test', *args, **kwargs)

    def configure_optimizers(self):
        params = [v for k, v in self.named_parameters() if 'ema' not in k]
        optim = get_optimizer(params, **self.hparams.optimizer)
        sched = get_scheduler(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}
