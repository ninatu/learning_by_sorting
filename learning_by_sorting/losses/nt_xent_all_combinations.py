import torch.nn as nn
from learning_by_sorting.losses.nt_xent import NT_XentWithMoreNegatives


class NT_XentAllCombinations(nn.Module):
    def __init__(self, n_augs, distributed=False, temperature=0.5,
                 n_pos_augs=None  # in multicrop strategy, only full crop images are used as positives,
                 # therefore, here "n_pos_augs" defines how many augs are used as positives
                 # (we store full crop in first augs: image_aug1, image_aug2, ...)
                 ):
        super().__init__()
        self.distributed = distributed
        self.similarity = nn.CosineSimilarity(dim=2)
        self.loss = NT_XentWithMoreNegatives(n_augs, temperature=temperature)
        self.temperature = temperature
        self.n_augs = n_augs
        self.n_pos_augs = n_pos_augs
        if self.n_pos_augs is None:
            self.n_pos_augs = n_augs

    def forward(self, data, output):
        final_loss_info = {}
        final_loss = 0
        total = 0

        for n in list(range(1, self.n_pos_augs + 1)): # one augmentation that is used as positive
            for m in range(n + 1, self.n_augs + 1): # another augmentation that is used as reference

                all_proj = {}
                for k in range(1, self.n_augs + 1):
                    all_proj[f'projection_image_aug{k}'] = output[f'projection_image_aug{k}']

                # reorder augmentations
                newoutput = {
                    'projection_image_aug1': all_proj.pop(f'projection_image_aug{n}'),
                    'projection_image_aug2': all_proj.pop(f'projection_image_aug{m}'),
                }
                for k, val in enumerate(all_proj.values()):
                    newoutput[f'projection_image_aug{3 + k}'] = val

                loss, loss_info = self.loss(data, newoutput)
                final_loss += loss
                total += 1

                suffix = '' if total == 1 else f'_comb{total}'
                for key, value in loss_info.items():
                    final_loss_info[key + suffix] = value

        final_loss = final_loss / total

        return final_loss, final_loss_info
