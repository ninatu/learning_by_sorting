import os
import json
import webdataset as wds
import tqdm
import random
import argparse

from torchvision.datasets.imagenet import ImageNet


MAXCOUNT_TRAIN = 2000
MAXCOUNT_TEST = 500
TRAINDATASET_SIZE = 1280585
N_TRAIN_RESHUFFLE = 4

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output', default='datasets/imagenet/processed/')

args = parser.parse_args()

original_dataset_root = args.input
processed_dataset_root = args.output


# https://github.com/tmbdev-archive/webdataset-examples/blob/master/makeshards.py
def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()

# https://github.com/tmbdev-archive/webdataset-examples/blob/master/makeshards.py
def write_val_dataset(dataset, maxcount, root, split):
    # We're using the torchvision ImageNet dataset
    # to parse the metadata; however, we will read
    # the compressed images directly from disk (to
    # avoid having to reencode them)
    ds = dataset
    nimages = len(ds.imgs)
    print("# nimages", nimages)
    # We shuffle the indexes to make sure that we
    # don't get any large sequences of a single class
    # in the dataset.
    indexes = list(range(nimages))
    random.shuffle(indexes)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(root, f"imagenet-{split}-%06d.tar")
    all_keys = set()

    os.makedirs(root, exist_ok=True)
    with wds.ShardWriter(pattern, maxcount=int(maxcount)) as sink:
        for i in indexes:
            # Internal information from the ImageNet dataset
            # instance: the file name and the numerical class.
            fname, cls = ds.imgs[i]
            assert cls == ds.targets[i]

            # Read the JPEG-compressed image file contents.
            image = readfile(fname)

            # Construct a uniqu keye from the filename.
            key = os.path.splitext(os.path.basename(fname))[0]
            wnid = os.path.basename(os.path.dirname(fname))

            # Useful check.
            assert key not in all_keys
            all_keys.add(key)

            # Construct a sample.
            xkey = key
            sample = {"__key__": xkey,
                      "image.jpg": image,
                      "metadata.pyd": {
                          'cls': cls,
                          'wnid': wnid,
                      }}

            # Write the sample to the sharded tar archives.
            sink.write(sample)

def write_train_dataset(dataset, maxcount, root, split):
    os.makedirs(root, exist_ok=True)
    pattern = os.path.join(root, f"imagenet-{split}-%06d.tar")
    all_keys = set()

    with wds.ShardWriter(pattern, maxcount=int(maxcount)) as sink:
        for key, image in tqdm.tqdm(iter(dataset), total=TRAINDATASET_SIZE):
            # Useful check.
            assert key not in all_keys
            all_keys.add(key)

            wnid = key.split('_')[0]
            cls = metadata['wnid_to_idx'][wnid]
            sample = {"__key__": key,
                      "image.jpg": image,
                      "metadata.pyd": {
                          'cls': cls,
                          'wnid': wnid,
                      }}
            # print(cls)
            sink.write(sample)


os.makedirs(processed_dataset_root, exist_ok=True)


# ----------------------------------- Get and save metadata -----------------------------------

dataset = ImageNet(original_dataset_root, split='val')
metadata = {
    'wnids': dataset.wnids,
    'classes': dataset.classes,
    'wnid_to_idx': dataset.wnid_to_idx,
    'class_to_idx': dataset.class_to_idx
}

os.makedirs(processed_dataset_root, exist_ok=True)

with open(os.path.join(processed_dataset_root, 'metadata.json'), 'w') as fout:
    json.dump(metadata, fout)

# ----------------------------------- Create shards for validation set -----------------------------------

shards_val_root = os.path.join(processed_dataset_root, 'val')
write_val_dataset(dataset, maxcount=MAXCOUNT_TEST, root=shards_val_root, split='val')


# ----------------------------------- Create shards for train set -----------------------------------

shards_train_root = os.path.join(processed_dataset_root, 'train')
dataset = ImageNet(original_dataset_root, split='train')
write_val_dataset(dataset, maxcount=MAXCOUNT_TRAIN, root=shards_val_root, split='val')
