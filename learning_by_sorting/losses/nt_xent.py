import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from learning_by_sorting.losses.utils import GatherLayer


class NT_Xent(nn.Module):
    def __init__(self, distributed=False, temperature=0.5,
                 stop_grad=False, topk_neg=None):
        super().__init__()
        self.distributed = distributed
        self.similarity = nn.CosineSimilarity(dim=2)
        self.temperature = temperature
        self.stop_grad = stop_grad
        self.topk_neg = topk_neg

    def forward(self, data, output):
        loss_info = {}

        z_i = output[f'projection_image_aug1'].float() # compute loss in full precision
        z_j = output[f'projection_image_aug2'].float()

        N = z_i.size(0)
        device = z_i.device

        z = torch.cat((z_i, z_j), dim=0)
        if self.distributed:
            z_all = torch.cat(GatherLayer.apply(z), dim=0)
        else:
            z_all = z

        if self.stop_grad:
            z_all = z_all.detach()

        sim = self.similarity(z.unsqueeze(1), z_all.unsqueeze(0)) / self.temperature

        # define positive pairs
        targets = torch.cat([
            torch.cat([torch.zeros((N, N)), torch.eye(N)], dim=1),
            torch.cat([torch.eye(N), torch.zeros((N, N))], dim=1),
        ], dim=0).to(device)

        # discard diagonal values
        mask = torch.eye(2 * N, dtype=torch.bool).to(device)

        if self.distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            pref_zeros = torch.zeros(2 * N, (2 * N) * rank, device=device)
            suf_zeros = torch.zeros(2 * N, 2 * N * (world_size - rank - 1), device=device)
            targets = torch.cat((pref_zeros, targets, suf_zeros), dim=1)
            mask = torch.cat((pref_zeros, mask, suf_zeros), dim=1)

        mask = mask > 0
        sim = sim[~mask].view(2 * N, -1)
        targets = targets[~mask].view(2 * N, -1)

        if self.topk_neg is not None:
            targets = targets > 0
            pos = sim[targets].view(2 * N, -1)
            negs = sim[~targets].view(2 * N, -1)
            negs, _ = torch.topk(negs, k=self.topk_neg, dim=1)

            sim = torch.cat((pos, negs), dim=1)
            targets = torch.zeros(2 * N, dtype=torch.long).to(device)

            if random.random() < 0.01:
                print(sim[0])

        else:
            targets = targets.argmax(dim=1)

        simclr_loss = F.cross_entropy(sim, targets, reduction='sum') / (2 * N)

        loss_info['simclr'] = simclr_loss.item()

        return simclr_loss, loss_info


class NT_XentWithMoreNegatives(nn.Module):
    def __init__(self, n_augs, distributed=False, temperature=0.5):
        super().__init__()
        self.n_augs = n_augs
        self.distributed = distributed
        self.similarity = nn.CosineSimilarity(dim=2)
        self.temperature = temperature

    def forward(self, data, output):
        loss_info = {}

        zs = []
        for n_aug in range(1, self.n_augs + 1):
            zs.append(output[f'projection_image_aug{n_aug}'].float()) # compute loss in full precision

        z_i = zs[0]
        z_j = zs[1]

        N = z_i.size(0)
        k = len(zs)
        device = z_i.device

        z = torch.cat((z_i, z_j), dim=0)
        z_all = torch.cat(zs, dim=0)
        if self.distributed:
            z_all = torch.cat(GatherLayer.apply(z_all), dim=0)
        sim = self.similarity(z.unsqueeze(1), z_all.unsqueeze(0)) / self.temperature

        # define positive pairs
        targets = torch.cat([
            torch.cat([torch.zeros((N, N)), torch.eye(N)] + [torch.zeros((N, N))] * (k - 2), dim=1),
            torch.cat([torch.eye(N), torch.zeros((N, N))] + [torch.zeros((N, N))] * (k - 2), dim=1),
        ], dim=0).to(device)

        # discard diagonal values
        mask = torch.cat([
            torch.cat([torch.eye(N), torch.zeros((N, N))] + [torch.eye(N)] * (k - 2), dim=1),
            torch.cat([torch.zeros((N, N)), torch.eye(N)] + [torch.eye(N)] * (k - 2), dim=1),
        ], dim=0).to(device)

        if self.distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            pref_zeros = torch.zeros(2 * N, (2 * N) * rank, device=device)
            suf_zeros = torch.zeros(2 * N, 2 * N * (world_size - rank - 1), device=device)
            targets = torch.cat((pref_zeros, targets, suf_zeros), dim=1)
            mask = torch.cat((pref_zeros, mask, suf_zeros), dim=1)

        mask = mask > 0
        sim = sim[~mask].view(2 * N, -1)
        targets = targets[~mask].view(2 * N, -1)
        targets = targets.argmax(dim=1)

        simclr_loss = F.cross_entropy(sim, targets, reduction='sum') / (2 * N)
        loss_info['simclr'] = simclr_loss.item()

        return simclr_loss, loss_info