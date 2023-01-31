import torch


def retrieval_mAP(rel):
    # rel: bool tensor which shape is (#.query, #.result)
    k = rel.shape[1]
    precs = rel.cumsum(dim=-1) / torch.arange(1, k + 1, device=rel.device)
    return ((rel * precs).sum(dim=-1) / rel.sum(dim=-1)).mean()


def retrieval_HR(rel, k: int = 1):
    # rel: bool tensor which shape is (#.query, #.result)
    k = min(rel.shape[1], k)
    return rel[:,:k].any(dim=-1).float().mean()


def retrieval_P(rel, k: int = 1):
    # rel: bool tensor which shape is (#.query, #.result)
    k = min(rel.shape[1], k)
    return rel[:,:k].sum(dim=-1).float().mean() / k


def retrieval_R(rel, k: int = 1):
    # rel: bool tensor which shape is (#.query, #.result)
    k = min(rel.shape[1], k)
    return torch.mean(rel[:,:k].sum(dim=-1) / rel.sum(dim=-1))


def retrieval_report(rel, metrics):
    report = {'mAP': retrieval_mAP(rel)}
    for m in metrics:
        n, k = m.split('.')
        if n == 'HR':
            v = retrieval_HR(rel, int(k))
        elif n == 'P':
            v = retrieval_P(rel, int(k))
        elif n == 'R':
            v = retrieval_R(rel, int(k))
        else:
            raise NotImplementedError(m)
        report[m] = v
    return report
