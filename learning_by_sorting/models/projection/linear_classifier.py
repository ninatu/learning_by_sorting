import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1000,
                 simsiam_initialization=True):
        super(LinearClassifier, self).__init__()

        fc = nn.Linear(input_dim, num_classes, bias=True)

        if simsiam_initialization:
            fc.weight.data.normal_(mean=0.0, std=0.01)
            fc.bias.data.zero_()

        self.projection = fc

    def train(self, mode=True):
        super().train(mode=mode)

    def forward(self, input):
        for key, value in list(input.items()):
            if 'encoder_image' in key:
                input[key.replace('encoder', 'projection')] = self.projection(value)

        return input
