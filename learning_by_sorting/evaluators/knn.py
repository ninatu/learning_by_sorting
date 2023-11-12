import os
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.nn import functional as F

from learning_by_sorting.trainer.ssl_trainer import _move_to_gpu


class KNNEvaluator:
    """
    Code is based on https://github.com/zhengyu-yang/lightning-bolts/blob/db6617f1ace068e467662c412e88947d28725e01/pl_bolts/callbacks/knn_online.py

    Weighted KNN online evaluator for self-supervised learning.
    The weighted KNN classifier matches sec 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
    The implementation follows:
        1. https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
        2. https://github.com/leftthomas/SimCLR
        3. https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    """

    def __init__(self, num_classes=1000, k=[1, 10, 20], temperature=0.07, use_weighting=True,
                 max_feature_bank=None,
                 distributed=False,
                 apply_evaluator_per_epoch=1,
                 compute_sim_on_cpu=False,
                 mute_tqdm=True,
                 log_name='',
                 ) -> None:
        """
        Args:
            k: k for k nearest neighbor
            temperature: temperature. See tau in section 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
        """
        # self.dataset: Optional[int] = None
        self.num_classes = num_classes
        self.k = k
        self.temperature = temperature
        self.distributed = distributed
        self.max_feature_bank = max_feature_bank
        self.apply_evaluator_per_epoch = apply_evaluator_per_epoch
        self.use_weighting = use_weighting
        self.compute_sim_on_cpu=compute_sim_on_cpu
        self.mute_tqdm = mute_tqdm
        self.log_name = log_name

    def compute_correct_preds(self, query_feature: Tensor, query_target: Tensor, feature_bank: Tensor, target_bank: Tensor):
        """
        Args:
            query_feature: (B, D) a batch of B query vectors with dim=D
            feature_bank: (N, D) the bank of N known vectors with dim=D
            target_bank: (N, ) the bank of N known vectors' labels

        Returns:
            (B, ) the predicted labels of B query vectors
        """

        B = query_feature.shape[0]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = query_feature @ feature_bank.T

        # [B, K]
        max_k = max(self.k)
        sim_weight, sim_indices = sim_matrix.topk(k=max_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(target_bank.expand(B, -1), dim=-1, index=sim_indices)

        n_correct_per_k = []

        for k in self.k:
            # counts for each class
            cur_sim_labels = sim_labels[:, :k]
            cur_sim_weight = sim_weight[:, :k]

            one_hot_label = torch.zeros(B * k, self.num_classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=cur_sim_labels.contiguous().view(-1, 1), value=1.0)

            if self.use_weighting:
                # weighted score ---> [B, C]
                cur_sim_weight = (cur_sim_weight / self.temperature).exp()
                pred_scores = torch.sum(one_hot_label.view(B, -1, self.num_classes) * cur_sim_weight.unsqueeze(dim=-1), dim=1)
            else:
                pred_scores = torch.sum(one_hot_label.view(B, -1, self.num_classes), dim=1)

            pred = pred_scores.argmax(dim=1)
            n_correct = (pred == query_target).sum().item()
            n_correct_per_k.append(n_correct)
        return n_correct_per_k

    @torch.no_grad()
    def validate(self, epoch, encoder, dataloaders_samplers, gpu, **kwargs):
        if ((epoch + 1) % self.apply_evaluator_per_epoch) != 0:
            return {}

        encoder.eval()

        (train_dataloader, train_datasampler), (valid_dataloader, valid_datasampler) = dataloaders_samplers
        if train_datasampler is None:
            os.environ["WDS_EPOCH"] = str(epoch)
        else:
            train_datasampler.set_epoch(epoch)
        if valid_datasampler is None:
            os.environ["WDS_EPOCH"] = str(epoch)
        else:
            valid_datasampler.set_epoch(epoch)

        # go through train data to generate feature bank
        print('=> KNN: compute train features', flush=True)

        feature_bank, target_bank = [], []
        feature_bank_size = 0
        for batch in tqdm.tqdm(train_dataloader, disable=self.mute_tqdm):
            batch = {key: _move_to_gpu(value, gpu) for key, value in batch.items()}
            output = encoder(batch)
            feature = output['encoder_image_aug1']
            target = batch['cls']

            feature = feature.flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            if self.compute_sim_on_cpu:
                feature = feature.cpu()
                target = target.cpu()

            feature_bank.append(feature)
            target_bank.append(target)
            feature_bank_size += feature.shape[0]

            if (self.max_feature_bank is not None) and (feature_bank_size > self.max_feature_bank):
                break

        feature_bank = torch.cat(feature_bank, dim=0) # [N, D]
        target_bank = torch.cat(target_bank, dim=0) # [N]
        if self.max_feature_bank is not None:
            feature_bank = feature_bank[:self.max_feature_bank]
            target_bank = target_bank[:self.max_feature_bank]

        print(f'=> KNN: feature_bank shape: {feature_bank.shape}', flush=True)

        # go through val data to predict the label by weighted knn search
        print('=> KNN: predict labels for val data', flush=True)

        total_correct_per_k = np.zeros(len(self.k))
        total_num = 0

        for batch in tqdm.tqdm(valid_dataloader, disable=self.mute_tqdm):
            batch = {key: _move_to_gpu(value, gpu) for key, value in batch.items()}
            output = encoder(batch)

            feature = output['encoder_image_aug1']
            target = batch['cls']

            feature = feature.flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            if self.compute_sim_on_cpu:
                feature = feature.cpu()
                target = target.cpu()

            n_correct_per_k = self.compute_correct_preds(feature, target, feature_bank, target_bank)

            total_correct_per_k += n_correct_per_k
            total_num += feature.shape[0]

        output = {
            f'knn@{k}{self.log_name}': total_correct / total_num
            for total_correct, k in zip(total_correct_per_k, self.k)
        }

        print(f'knn@{self.k}{self.log_name} in %:', *[f'{total_correct / total_num  * 100:.1f}' for total_correct in total_correct_per_k], flush=True)

        return output
