import webdataset as wds
from functools import partial
import torch
from .utils import get_transforms, preprocess_sample
from learning_by_sorting.utils.dist_utils import get_world_size, get_rank


def web_dataloader(shards, dataset_size, shuffle, batch_size, num_workers, train=True, distributed=True,
                   transform_type='val', n_augs=1):
    if distributed:
        print(f'=> adapting batch_size with respect to {get_world_size()} workers', flush=True)
        batch_size = int(batch_size / get_world_size())
        print(f'=> adapting dataset_size with respect to {get_world_size()} workers', flush=True)
        dataset_size = int(dataset_size / get_world_size())

    transforms = get_transforms(transform_type=transform_type, n_augs=n_augs)
    preprocess_func = partial(preprocess_sample, transforms=transforms, n_augs=n_augs)
    shardlist = wds.PytorchShardList(shards, epoch_shuffle=True)
    train_dataset = (
        wds.WebDataset(shardlist)
            .shuffle(shuffle)
            .decode("pil")
            .map(preprocess_func)
    )

    number_of_batches = dataset_size // batch_size

    torch.multiprocessing.set_start_method('spawn', force=True)
    dataloader = wds.WebLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=train, pin_memory=False,
        multiprocessing_context='spawn' if num_workers > 0 else None) \
        .with_length(number_of_batches)

    sampler = None
    return dataloader, sampler
