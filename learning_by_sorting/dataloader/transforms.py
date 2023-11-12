import random
import numpy as np
from PIL import ImageFilter, ImageOps
from PIL import Image

from torchvision import transforms as transforms


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianBlurDINO(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_simclr_aug_transform(size=224):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
    ])
    return transform


def get_dino_aug_transforms(size=224, n_augs=None, scale=(0.14, 1.0)):
    global_transfo1 = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlurDINO(1.0),
    ])
    # second global crop
    global_transfo2 = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlurDINO(0.1),
        Solarization(0.2),
    ])
    if n_augs is None or n_augs == 2:
        return [global_transfo1, global_transfo2]
    elif n_augs == 3:
        return [global_transfo1, global_transfo2, global_transfo1]
    elif n_augs == 4:
        return [global_transfo1, global_transfo2, global_transfo1, global_transfo2]
    else:
        raise NotImplementedError()


def get_dino_aug_multicrop_transforms(size_crops=[224, 96], scale_small_crops=(0.05, 0.14), scale_big_crops=(0.14, 1.0)):
    trans = get_dino_aug_transforms(size_crops[0], scale=scale_big_crops)
    trans.extend([transforms.Compose([
        transforms.RandomResizedCrop(size_crops[1], scale=scale_small_crops, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlurDINO(p=0.5),
    ])] * 6)
    return trans
