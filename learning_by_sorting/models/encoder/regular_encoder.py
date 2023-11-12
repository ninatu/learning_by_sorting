import torch.nn
import torch.nn as nn
import torchvision.models as models
from functools import partial


class RegularEncoder(nn.Module):
    def __init__(self, arch):
        super(RegularEncoder, self).__init__()

        base_encoder = partial(models.__dict__[arch], zero_init_residual=True)
        self.encoder = base_encoder()

        prev_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()

        self.feature_dim = prev_dim

    def forward(self, input):
        output = {}
        for key, value in input.items():
            if 'image' in key:
                z = self.encoder(value)
                output['encoder_' + key] = z
        input.update(output)
        return input
