import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from learning_by_sorting.dataloader.utils import preprocess_sample, get_transforms
from learning_by_sorting.utils.dist_utils import get_world_size, get_rank


class SSLImageNet(ImageNet):
    def __init__(self, root, split, n_augs=1, transform=None):
        super(SSLImageNet, self).__init__(root, split)
        self.n_augs = n_augs
        self.ssl_transforms = transform

    def __getitem__(self, index):
        image, cls = super().__getitem__(index)
        sample = {
            'image.jpg': image,
            'metadata.pyd': {'cls': cls}
        }
        output = preprocess_sample(sample, self.ssl_transforms, n_augs=self.n_augs)
        return output


def image_folder_dataloader(root, shuffle, batch_size, num_workers, train=True, distributed=True,
                   transform_type='val', n_augs=1):
    transforms = get_transforms(transform_type=transform_type, n_augs=n_augs)
    dataset = SSLImageNet(root, split=('train' if train else 'val'), transform=transforms, n_augs=n_augs)

    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(),
                                                      shuffle=shuffle)
        shuffle = False
        print(f'=> adapting batch_size with respect to {get_world_size()} workers', flush=True)
        batch_size = int(batch_size / get_world_size())
    else:
        shuffle = True
        sampler = None

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=False,
                            sampler=sampler,
                            shuffle=shuffle,
                            drop_last=train,
                            )

    return dataloader, sampler