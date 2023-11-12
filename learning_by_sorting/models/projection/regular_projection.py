from torch import nn as nn


class RegularProjection(nn.Module):
    def __init__(self, input_dim, num_layers=3, output_dim=2048, batch_norm_first=False):
        super(RegularProjection, self).__init__()

        self.projection = get_projection_head(num_layers, input_dim=input_dim, output_dim=output_dim,
                                              batch_norm_first=batch_norm_first)

    def forward(self, input):
        output = {}
        for key, value in list(input.items()):
            if 'encoder_image' in key:
                output[key.replace('encoder', 'projection')] = self.projection(value)
        input.update(output)

        return input


def get_projection_head(num_layers, input_dim, output_dim, batch_norm_first=False):
    layers = []
    if batch_norm_first:
        layers.append(nn.BatchNorm1d(input_dim))

    for n in range(num_layers - 1):
        layers += [nn.Linear(input_dim, input_dim, bias=False),
                   nn.BatchNorm1d(input_dim),
                   nn.ReLU(inplace=True)]

    layers += [nn.Linear(input_dim, output_dim, bias=True),
               nn.BatchNorm1d(output_dim, affine=False)]

    # for comparability
    layers[-2].bias.requires_grad = False

    return nn.Sequential(*layers)