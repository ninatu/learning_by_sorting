import random
import torch.nn as nn
import diffsort
import torch
import torch.distributed as dist
import torch.nn.functional as F

from learning_by_sorting.losses.utils import GatherLayer, CosineSimilarityAlltoAll


class GroCo(nn.Module):
    def __init__(self,
                 distributed=False,
                 steepness=1,
                 n_augs=2,
                 topk_neg=10,
                 stop_grad=True,
                 n_pos_augs=None, # in multicrop strategy, only full crop images are used as positives,
                                  # therefore, here "n_pos_augs" defines how many augs are used as positives
                                  # (we store full crop in first augs: image_aug1, image_aug2, ...)
                 ):
        super(GroCo, self).__init__()
        assert topk_neg >= 0
        self.distributed = distributed
        self.topk_neg = topk_neg
        self.stop_grad = stop_grad

        self.n_augs = n_augs
        self.n_pos_augs = n_pos_augs
        if self.n_pos_augs is None:
            self.n_pos_augs = n_augs

        sorting_network_type = 'odd_even'
        interpolation_type = 'cauchy'
        art_lambda = 0.25

        num_compare = (self.n_pos_augs - 1) + topk_neg  # we have (n_pos_augs - 1) positives and topk_neg negative
        self.sorter = diffsort.DiffSortNet(
            sorting_network_type=sorting_network_type,
            interpolation_type=interpolation_type,
            size=num_compare,
            device='cpu',
            steepness=steepness,
            art_lambda=art_lambda,
        )

        num_compare = self.n_pos_augs + topk_neg  # in multicrop setup, for small crops, we would have n_pos_augs positives
        self.sorter2 = diffsort.DiffSortNet(
            sorting_network_type=sorting_network_type,
            interpolation_type=interpolation_type,
            size=num_compare,
            device='cpu',
            steepness=steepness,
            art_lambda=art_lambda,
        )

        self.pairwise_similarity = nn.CosineSimilarity(dim=1)
        self.alltoall_similarity = CosineSimilarityAlltoAll()
        self.loss = torch.nn.BCELoss()

    def cuda(self, device=None):
        super(GroCo, self).cuda(device)
        self.sorter.sorting_network = [[matrix.cuda(device) for matrix in matrix_set]
                                                    for matrix_set in self.sorter.sorting_network]
        self.sorter2.sorting_network = [[matrix.cuda(device) for matrix in matrix_set]
                                                    for matrix_set in self.sorter2.sorting_network]
        return self

    def to(self, device=None):
        super(GroCo, self).to(device)
        self.sorter.sorting_network = [[matrix.to(device) for matrix in matrix_set]
                                       for matrix_set in self.sorter.sorting_network]
        self.sorter2.sorting_network = [[matrix.to(device) for matrix in matrix_set]
                                       for matrix_set in self.sorter2.sorting_network]
        return self

    def forward(self, data, output):
        final_loss_info = {}
        final_loss = 0

        positives = [f'projection_image_aug{i}' for i in range(1, 1 + self.n_pos_augs)]
        for n in range(1, 1 + self.n_augs):
            anchor = f'projection_image_aug{n}'
            cur_positives = [p for p in positives if p != anchor] # exclude anchor
            loss, loss_info = self.compute_loss(output, anchor, cur_positives)
            final_loss += loss

            if n == 1:  # save to the log once
                for key, value in loss_info.items():
                    final_loss_info[key] = value

        final_loss = final_loss / self.n_augs

        return final_loss, final_loss_info

    def _get_positive_mask(self, device, bs_per_gpu, bs):
        diagonal = torch.eye(bs_per_gpu, device=device)
        if self.distributed:
            rank = dist.get_rank()
            pref_zeros = torch.zeros(bs_per_gpu, bs_per_gpu * rank, device=device)
            suf_zeros = torch.zeros(bs_per_gpu, bs - bs_per_gpu * (rank + 1), device=device)
            diagonal = torch.cat((pref_zeros, diagonal, suf_zeros), dim=1)
        diagonal = diagonal > 0
        pos_mask = diagonal
        return pos_mask

    def compute_loss(self, output, anchor_name, positive_names):
        loss_info = {}

        anchor = output[anchor_name].float() # compute loss in full precision

        # compute similarities to the positives
        pos_sims = []
        for positive_name in positive_names:
            positive = output[positive_name].float() # compute loss in full precision
            pos_sims.append(self.pairwise_similarity(anchor, positive.detach() if self.stop_grad else positive))
        pos_sims = torch.stack(pos_sims, dim=1)  # B, n_positives

        # compute similarities to the negatives
        if self.distributed:
            negatives = torch.cat(GatherLayer.apply(anchor), dim=0)

            device = negatives.device
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            bs_per_gpu = len(anchor)
            non_negatives_mask = torch.eye(bs_per_gpu, device=device)
            pref_zeros = torch.zeros(bs_per_gpu, bs_per_gpu * rank, device=device)
            suf_zeros = torch.zeros(bs_per_gpu, bs_per_gpu * world_size - bs_per_gpu * (rank + 1), device=device)
            non_negatives_mask = torch.cat((pref_zeros, non_negatives_mask, suf_zeros), dim=1)
            non_negatives_mask = non_negatives_mask > 0
        else:
            negatives = anchor
            non_negatives_mask = torch.eye(len(anchor), device=anchor.device) > 0

        neg_sims = self.alltoall_similarity(anchor, negatives.detach() if self.stop_grad else negatives)
        neg_sims = neg_sims.masked_fill(non_negatives_mask, -float("inf"))

        # preordering
        pos_sims, _ = torch.sort(pos_sims, dim=1)  # make asсending order
        neg_sims, _ = torch.topk(neg_sims, k=self.topk_neg, dim=1)
        neg_sims = torch.flip(neg_sims, dims=[1])  # make asсending order
        sims = torch.cat((neg_sims, pos_sims), dim=1)

        # logging
        loss_info['info_pos_max'] = pos_sims[:, -1].mean().item()
        loss_info['info_pos_min'] = pos_sims[:, 0].mean().item()
        loss_info['info_neg_max'] = neg_sims[:, -1].mean().item()
        loss_info['info_neg_min'] = neg_sims[:, 0].mean().item()
        loss_info['info_pos_neg_dist'] = loss_info['info_pos_min'] - loss_info['info_neg_max']
        if random.random() < 0.01:
            print("GroCo Loss: example to sort: ", sims[0])

        if pos_sims.size(1) == self.n_pos_augs - 1:
            _, perm_prediction = self.sorter(sims)
        else:
            _, perm_prediction = self.sorter2(sims)

        targets = torch.tensor(list(range(sims.shape[1])))[None].repeat(sims.shape[0], 1).to(sims.device)  # from smallest to largest
        perm_target = torch.nn.functional.one_hot(torch.argsort(targets, dim=-1)).transpose(-2, -1).float()

        # sum probabilities for positive ranks
        perm_prediction_neg = perm_prediction[:, :, :self.topk_neg].sum(dim=2, keepdim=True)
        perm_prediction_pos = perm_prediction[:, :, self.topk_neg:].sum(dim=2, keepdim=True)

        # sum probabilities for negative ranks
        perm_target_neg = perm_target[:, :, :self.topk_neg].sum(dim=2, keepdim=True)
        perm_target_pos = perm_target[:, :, self.topk_neg:].sum(dim=2, keepdim=True)

        # clip (due to numerical errors)
        perm_prediction_neg = torch.clip(perm_prediction_neg, 0, 1)
        perm_prediction_pos = torch.clip(perm_prediction_pos, 0, 1)
        perm_target_neg = torch.clip(perm_target_neg, 0, 1)
        perm_target_pos = torch.clip(perm_target_pos, 0, 1)

        sorting_loss = (self.loss(perm_prediction_pos, perm_target_pos) + self.loss(perm_prediction_neg, perm_target_neg)) / 2

        loss_info['sorting'] = sorting_loss.item()

        return sorting_loss, loss_info
