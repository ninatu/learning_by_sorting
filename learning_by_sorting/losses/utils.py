import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CosineSimilarityAlltoAll(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        return cosine_similatity(x, y)


def _normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


def cosine_similatity(a, b, eps=1e-8, flag_normalize_embeddings=True):
    """
    added eps for numerical stability
    """
    if flag_normalize_embeddings:
        a = _normalize_embeddings(a, eps)
        b = _normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt