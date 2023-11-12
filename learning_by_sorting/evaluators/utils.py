import torch
import torch.distributed as dist

from learning_by_sorting.trainer.ssl_trainer import _move_to_gpu


def accuracy(output, target):
    with torch.no_grad():
        _, pred = output.topk(10, 1, True, True)
        correct = pred.eq(target.reshape(-1, 1).expand_as(pred))

        total_top_1 = correct[:, :1].sum()
        total_top_5 = correct[:, :5].sum()
        total_top_10 = correct[:, :10].sum()

        return total_top_1, total_top_5, total_top_10


def all_reduce_multigpu(values, gpu):
    values = [_move_to_gpu(torch.tensor(values), gpu)]
    dist.all_reduce_multigpu(values)

    return values[0].cpu().numpy().tolist()