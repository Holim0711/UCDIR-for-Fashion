import torch

__all__ = ['BNN', 'CGD']


class BNN(torch.nn.Module):
    """ Batch Normalization Neck: https://arxiv.org/pdf/1906.08332.pdf """
    def __init__(self, num_features=2048):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.bn = torch.nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        aux = x
        x = self.bn(x)
        return x, aux


def gem_pooling2d(x, p):
    x = x.pow(p)
    x = torch.nn.functional.avg_pool2d(x, x.shape[-2:])
    return x.pow(1 / p)


class GeMPool2d(torch.nn.Module):
    def __init__(self, p: int = 3):
        super().__init__()
        self.register_buffer('p', torch.tensor(p))

    def forward(self, x):
        return gem_pooling2d(x, self.p)


class CGD(torch.nn.Module):
    """ Combo of Global Descriptors: https://arxiv.org/pdf/1903.10663.pdf """
    def __init__(self, in_features=2048, out_features=1536, config='SMG'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        dim = out_features // len(config)
        self.poolings = torch.nn.ModuleList([
            torch.nn.AdaptiveAvgPool2d(1) if c == 'S' else
            torch.nn.AdaptiveMaxPool2d(1) if c == 'M' else
            GeMPool2d()
            for c in config
        ])
        self.reductions = torch.nn.ModuleList([
            torch.nn.Linear(in_features, dim)
            for _ in config
        ])
        self.fc = torch.nn.Linear(dim * len(config), out_features, bias=False)

    def forward(self, x):
        desc = [p(x) for p in self.poolings]
        desc = [torch.flatten(d, 1) for d in desc]
        aux = desc[0]
        desc = [r(d) for d, r in zip(desc, self.reductions)]
        desc = [torch.nn.functional.normalize(d) for d in desc]
        feat = self.fc(torch.concat(desc))
        return feat, aux
