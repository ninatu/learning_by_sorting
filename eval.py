import argparse
import random
import warnings
import collections
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from learning_by_sorting.utils.dist_utils import init_distributed_mode, is_main_process, get_rank
from learning_by_sorting.utils.parse_config import ConfigParser
from learning_by_sorting.trainer import SSLTrainer
import learning_by_sorting.dataloader as module_dataloader
import learning_by_sorting.models.encoder as module_encoder
import learning_by_sorting.evaluators as module_evaluators

from sacred import Experiment
from neptunecontrib.monitoring.sacred import NeptuneObserver


ex = Experiment('train', save_git_info=False)


@ex.main
def train():
    if config['seed'] is not None:
        # fix random seeds for reproducibility
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        seed = config['seed'] + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    args = config.args

    # create model
    print("=> creating model", flush=True)
    encoder = config.initialize('encoder', module_encoder)

    print(encoder)

    if args.distributed:
        # Apply SyncBN
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        torch.cuda.set_device(args.gpu)
        encoder.cuda(args.gpu)
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        encoder.cuda(args.gpu)
    else:
        device = torch.device(args.device)
        encoder.to(device)

    if isinstance(config['valid_dataloader'], list):
        valid_dataloader_sampler = [config.initialize('valid_dataloader', module_dataloader, index=i, distributed=args.distributed)
                         for i in range(len(config['valid_dataloader']))]
    else:
        valid_dataloader_sampler = config.initialize('valid_dataloader', module_dataloader, distributed=args.distributed)

    if isinstance(config['evaluator'], list):
        evaluator = [config.initialize('evaluator', module_evaluators, index=i, distributed=args.distributed)
                     for i in range(len(config['evaluator']))]
    else:
        evaluator = config.initialize('evaluator', module_dataloader, distributed=args.distributed)

    trainer = SSLTrainer(encoder=encoder,
                         projection=None,
                         loss=None,
                         optimizer=None,
                         lr_scheduler=None,
                         config=config,
                         train_data_loader=None,
                         train_data_sampler=None,
                         gpu=args.gpu,
                         valid_dataloader_sampler=valid_dataloader_sampler,
                         evaluator=evaluator,
                         scheduler_update='iter',
                         writer=ex if is_main_process() else None,
                         is_main_trainer=is_main_process(),
                         init_val=True,
                         resume_only_encoder=True,
                         epochs=0,
                         mixed_precision=False,
                         save_latest=False,
                         )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-n', '--neptune', action='store_true', help='whether to observe (neptune)')
    parser.add_argument('--name', default=None, type=str, help='optional: specify experiment name (default: None)')

    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    # custom cli options to modify configuration from default values given in json file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--init_val'], type=int, target=('trainer', 'init_val')),
    ]
    config = ConfigParser(parser, options)
    ex.add_config(config.config)
    args = config.args

    init_distributed_mode(args=args)
    if not args.distributed:
        if args.device == 'cuda' and args.gpu is None:
            args.gpu = 0

    args = config.args
    print(f"=> running {args.config}", flush=True)

    if args.neptune and is_main_process():
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError

        ex.observers.append(NeptuneObserver(
            project_name="",
            api_token="",
            source_extensions=['train.py', 'eval.py', 'learning_by_sorting/**/*.py', 'configs/**/*.yaml', 'configs/**/*.yml']))

    ex.run()
