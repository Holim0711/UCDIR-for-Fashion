import torch
from torchvision.models import resnet50
import pytorch_lightning as pl
from weaver import get_optimizer, get_scheduler


def load_moco_v2():
    model = resnet50()
    model.fc = torch.nn.Identity()
    checkpoint = torch.load('misc/moco_v2_800ep_pretrain.pth.tar')
    prefix = 'module.encoder_q.'
    state_dict = checkpoint['state_dict']
    model.load_state_dict({
        k: state_dict[prefix + k] for k in model.state_dict()
        if (prefix + k) in state_dict
    })
    return model


def change_bn_momentum(model: torch.nn.Module, momentum: float):
    if isinstance(model, torch.nn.BatchNorm2d):
        model.momentum = 1 - momentum
    for child in model.children():
        change_bn_momentum(child, momentum)


class EMAModel(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model: torch.nn.Module, a: float):
        super().__init__(model, avg_fn=lambda m, x, _: a * m + (1 - a) * x)

    def update_parameters(self, model):
        super().update_parameters(model)
        # BatchNorm buffers are already EMA
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, bias=False),
            # torch.nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return torch.nn.functional.normalize(x)


def pairwise_cosine_similarity(x1, x2, eps=1e-8):
    w1 = x1.norm(dim=-1, keepdim=True)
    w2 = x2.norm(dim=-1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def retrieval_mAP(rels):
    k = rels.shape[1]
    precs = rels.cumsum(dim=-1) / torch.arange(1, k + 1, device=rels.device)
    return ((rels * precs).sum(dim=-1) / rels.sum(dim=-1)).mean()


def retrieval_HR(rels, k=1):
    k = min(rels.shape[1], k)
    return rels[:,:k].any(dim=-1).float().mean()


def retrieval_P(rels, k=1):
    k = min(rels.shape[1], k)
    return rels[:,:k].sum(dim=-1).float().mean() / k


class BaseModule(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = load_moco_v2()
        self.head = ProjectionHead(2048, 128, 128)

        change_bn_momentum(self.model, self.hparams.ema)
        self.ema_model = EMAModel(self.model, self.hparams.ema)
        change_bn_momentum(self.head, self.hparams.ema)
        self.ema_head = EMAModel(self.head, self.hparams.ema)

    def forward(self, x):
        return self.ema_model(x)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
        self.ema_head.update_parameters(self.head)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, c = batch
        return {'v': self(x), 'c': c}

    def validation_epoch_end(self, outputs):
        sources, targets = outputs
        sources = {
            'v': torch.concat([x['v'] for x in sources]),
            'c': torch.concat([x['c'] for x in sources]),
        }
        targets = {
            'v': torch.concat([x['v'] for x in targets]),
            'c': torch.concat([x['c'] for x in targets]),
        }

        if self.trainer.world_size > 1:
            targets = self.all_gather(targets)
            targets = {k: v.flatten(0, 1) for k, v in targets.items()}

        simmat = pairwise_cosine_similarity(sources['v'], targets['v'])
        preds = targets['c'][simmat.argsort(descending=True)]
        rels = (preds == sources['c'].unsqueeze(-1))

        mAP = retrieval_mAP(rels)
        self.log('val/mAP', mAP, sync_dist=True)

        for metric in self.hparams.dataset['metric']:
            name, k = metric.split('@')
            if name == 'HR':
                v = retrieval_HR(rels, int(k))
            elif name == 'P':
                v = retrieval_P(rels, int(k))
            else:
                raise NotImplementedError(metric)
            self.log('val/' + metric.replace('@', '.'), v, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        params = self.model.parameters()
        optim = get_optimizer(params, **self.hparams.optimizer)
        sched = get_scheduler(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}