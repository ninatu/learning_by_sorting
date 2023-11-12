from torchvision import transforms as transforms

from learning_by_sorting.dataloader.transforms import get_simclr_aug_transform, get_dino_aug_transforms, \
    get_dino_aug_multicrop_transforms


def get_transforms(transform_type='val', n_augs=None):
    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if transform_type == 'val':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        return transforms.Compose([val_transform, post_transform])
    if transform_type == 'val_bicubic_inter':
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
        ])
        return transforms.Compose([val_transform, post_transform])
    elif transform_type == 'linear_prob':
        finetune_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        return transforms.Compose([finetune_transform, post_transform])
    elif transform_type == 'simclr':
        return transforms.Compose([get_simclr_aug_transform(), post_transform])
    elif transform_type == 'dino':
        ref_transforms = get_dino_aug_transforms(n_augs=n_augs)
        return [transforms.Compose([tr, post_transform])  for tr in ref_transforms]
    elif transform_type == 'dino_multicrop':
        ref_transforms = get_dino_aug_multicrop_transforms()
        assert n_augs == len(ref_transforms)
        return [transforms.Compose([tr, post_transform])  for tr in ref_transforms]
    else:
        raise NotImplementedError


def preprocess_sample(sample, transforms, n_augs=1):
    image = sample['image.jpg']
    cls = sample['metadata.pyd']['cls']

    if isinstance(transforms, list):
        transform_list = transforms
        assert len(transform_list) == n_augs
    else:
        transform_list = [transforms] * n_augs

    output = {}
    output['cls'] = cls
    for n, tr in enumerate(transform_list):
        output[f'image_aug{n + 1}'] = tr(image)

    return output