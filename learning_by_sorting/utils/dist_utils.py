import os
import torch
import torch.distributed as dist


# The code is based on https://github.com/salesforce/BLIP/blob/main/utils.py
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            print('Using distributed mode (RANK)')
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            print('Using distributed mode (SLURM_PROCID)')
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            print('Not using distributed mode')
            args.distributed = False
            return

        # torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        print('Distributed init (rank {}, world {}, gpu {}), url:{}'.format(
            args.rank, args.world_size, args.gpu, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)
    else:
        print('Not using distributed mode')


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
