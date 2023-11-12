import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, distributed=False):
        super(CrossEntropyLoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, data, output):
        loss = self.cross_entropy(output['projection_image_aug1'], data['cls'])
        loss_info = {
            'cross_entropy': loss.item()
        }
        return loss, loss_info