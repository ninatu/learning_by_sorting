import math


class CustomCosineSchedulerWithWarmup:
    def __init__(self, optimizer, init_lr, epochs, warmup_epochs=0, lr_min=0):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr_min = lr_min

    def step(self, epoch):
        epoch = epoch % self.epochs
        if epoch < self.warmup_epochs:
            cur_lr = epoch / self.warmup_epochs * self.init_lr
        else:
            cur_lr = self.lr_min + (self.init_lr - self.lr_min) * 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
