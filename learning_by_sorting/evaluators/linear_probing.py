import numpy as np
import torch
import os
import tqdm
import torch.distributed as dist

from learning_by_sorting.evaluators.utils import accuracy, all_reduce_multigpu
from learning_by_sorting.trainer.ssl_trainer import _move_to_gpu
from learning_by_sorting.utils.larc import LARC
from learning_by_sorting.models.projection import LinearClassifier
from learning_by_sorting.utils.scheduler import CustomCosineSchedulerWithWarmup
from learning_by_sorting.losses import CrossEntropyLoss
from learning_by_sorting.utils.dist_utils import get_world_size

class LinearProbingEvaluator:
    def __init__(self, distributed, hyperparams=None):
        # We can train many linear classifiers with different hyperparameters in parallel (with one forward pass of the encoder),
        # therefore we take multiple hyperparameter sets here

        hyperparams = hyperparams if isinstance(hyperparams, list) else [hyperparams]
        for hyperparam in hyperparams:
            assert 'log_name' in hyperparam
            assert 'apply_evaluator_after_epochs' in hyperparam
            assert 'epochs' in hyperparam
            assert 'base_learning_rate' in hyperparam
            assert 'batch_size' in hyperparam
            assert 'optimizer' in hyperparam
            assert 'optimizer_use_larc' in hyperparam
            assert 'lr_scheduler' in hyperparam
        self.hyperparams_list = hyperparams
        self.distributed = distributed

    def validate(self, epoch, encoder, dataloaders_samplers, gpu, logger=None, **kwargs):
        if not any([((hp['apply_evaluator_after_epochs'] is None) or
                  (epoch in hp['apply_evaluator_after_epochs']))
                 for hp in self.hyperparams_list]):
            return {}

        encoder.eval()

        (train_dataloader, train_datasampler), (valid_dataloader, valid_datasampler) = dataloaders_samplers
        if train_datasampler is None:
            os.environ["WDS_EPOCH"] = str(epoch)
        else:
            train_datasampler.set_epoch(epoch)
        if valid_datasampler is None:
            os.environ["WDS_EPOCH"] = str(epoch)
        else:
            valid_datasampler.set_epoch(epoch)

        # Precompute embeddings for validation
        print('=> Linear probing: precomputing features of val set ... ', flush=True)
        with torch.no_grad():
            val_embeddings, val_targets = [], []
            for batch in tqdm.tqdm(valid_dataloader):
                batch = {key: _move_to_gpu(value, gpu) for key, value in batch.items()}
                output = encoder(batch)

                val_embeddings.append(output['encoder_image_aug1'].detach().cpu())
                val_targets.append(batch['cls'])
            val_embeddings = torch.cat(val_embeddings, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
        val_embeddings = _move_to_gpu(val_embeddings, gpu)
        val_targets = _move_to_gpu(val_targets, gpu)

        # Create training objects (linear_classifier, optimizer, etc) for each hyperparam set
        training_objects_list = [create_training_objects(self.distributed, gpu, encoder, hp)
                                 for hp in self.hyperparams_list if (
                                         (hp['apply_evaluator_after_epochs'] is None) or
                                         (epoch in hp['apply_evaluator_after_epochs']))]

        # Train the linear classifiers
        max_epoch = max(training_objects['epochs'] for training_objects in training_objects_list)

        print('=> Linear probing: training ... ', flush=True)
        for cur_epoch in tqdm.tqdm(range(max_epoch)):
            os.environ["WDS_EPOCH"] = f'{cur_epoch}'

            for training_objects in training_objects_list:
                training_objects['lr_scheduler'].step(cur_epoch)
                training_objects['linear_classifier'].train()

            # Train one epoch
            for batch in tqdm.tqdm(train_dataloader):
                with torch.no_grad():
                    batch = {key: _move_to_gpu(value, gpu) for key, value in batch.items()}
                    encoder_output = encoder(batch)

                for training_objects in training_objects_list:
                    assert training_objects['batch_size'] == (len(encoder_output['encoder_image_aug1']) * get_world_size())

                    if training_objects['epochs'] <= cur_epoch:
                        continue

                    output = training_objects['linear_classifier'](encoder_output)
                    loss, loss_info = training_objects['criterion'](batch, output)

                    if logger is not None:
                        top_1, top_5, top_10 = accuracy(output['projection_image_aug1'], batch['cls'])
                        logger.log_scalar(f'extra_logging_linear_prob{training_objects["log_name"]}_after_{epoch}ep_train_top_1',
                                          top_1.item() / batch['cls'].shape[0])

                    # Compute gradient and do optimization step
                    training_objects['optimizer'].zero_grad()
                    loss.backward()
                    training_objects['optimizer'].step()

            # Evaluation after one epoch
            for training_objects in training_objects_list:
                if training_objects['epochs'] <= cur_epoch:
                    continue
                results = evaluate(val_embeddings, val_targets, training_objects['linear_classifier'],
                                   training_objects['criterion'], self.distributed, gpu, batch_size=4096)
                if logger is not None:
                    for key, value in results.items():
                        logger.log_scalar(f'extra_logging_linear_prob{training_objects["log_name"]}_after_{epoch}ep_val_{key}',
                                          value)

        # Evaluation
        output = {}
        for training_objects in training_objects_list:
            results = evaluate(val_embeddings, val_targets, training_objects['linear_classifier'], training_objects['criterion'], self.distributed, gpu, batch_size=4096)
            for key, value in results.items():
                output[f'linear_prob{training_objects["log_name"]}_val_{key}'] = value

        return output


def create_training_objects(distributed, gpu, encoder, hyperparams):
    if distributed:
        linear_classifier = LinearClassifier(input_dim=encoder.module.feature_dim)
    else:
        linear_classifier = LinearClassifier(input_dim=encoder.feature_dim)

    if distributed:
        linear_classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)
        linear_classifier.cuda()
        linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[gpu])
    elif gpu is not None:
        linear_classifier.cuda(gpu)

    parameters = list(linear_classifier.parameters())
    print('#Optimize parameters: ', len(parameters))
    assert len(parameters) == 2  # fc.weight, fc.bias

    init_lr = hyperparams['base_learning_rate'] * hyperparams['batch_size'] / 256

    if hyperparams['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=parameters, lr=init_lr, **hyperparams['optimizer']['args'])
    elif hyperparams['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=parameters, lr=init_lr, **hyperparams['optimizer']['args'])
    else:
        raise NotImplementedError

    if hyperparams['optimizer_use_larc']:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    if hyperparams['lr_scheduler']['type'] == 'CustomCosineSchedulerWithWarmup':
        lr_scheduler = CustomCosineSchedulerWithWarmup(optimizer=optimizer, init_lr=init_lr, **hyperparams['lr_scheduler']['args'])
    elif hyperparams['lr_scheduler']['type'] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, **hyperparams['lr_scheduler']['args'])
    else:
        raise NotImplementedError

    criterion = CrossEntropyLoss()

    return {
        'linear_classifier': linear_classifier,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'criterion': criterion,
        'epochs': hyperparams['epochs'],
        'log_name': hyperparams['log_name'],
        'batch_size': hyperparams['batch_size'],
    }


def evaluate(val_embeddings, val_targets, linear_classifier, criterion, distributed, gpu, batch_size=4096):
    linear_classifier.eval()
    total_top_1, total_top_5, total_top_10, total_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        effective_batch_size = batch_size // dist.get_world_size() if distributed else batch_size
        val_size = len(val_embeddings)
        n_iters = int(np.floor(val_size) / effective_batch_size)
        for iter in range(n_iters):
            data = val_embeddings[iter * effective_batch_size:(iter + 1) * effective_batch_size]
            target = val_targets[iter * effective_batch_size:(iter + 1) * effective_batch_size]

            output = linear_classifier({'encoder_image_aug1': data})
            loss, loss_info = criterion({'cls': target}, output)

            top_1, top_5, top_10 = accuracy(output['projection_image_aug1'], target)
            num = output['projection_image_aug1'].shape[0]

            total_loss += loss.item() * num
            total_top_1 += top_1
            total_top_5 += top_5
            total_top_10 += top_10
            total_num += num

    # gather representations from other gpus
    if distributed:
        total_num = all_reduce_multigpu([total_num], gpu)[0]
        total_loss, total_top_1, total_top_5, total_top_10 = all_reduce_multigpu([total_loss, total_top_1, total_top_5, total_top_10], gpu)

    output = {
        # 'val_loss': total_loss / total_num,
        'top_1': total_top_1 / total_num,
        'top_5': total_top_5 / total_num,
        # 'val_top_10': total_top_10 / total_num,
    }
    return output