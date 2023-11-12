import time
import torch
from torch.cuda.amp import autocast, GradScaler

from learning_by_sorting.base.base_trainer import BaseTrainer
from learning_by_sorting.utils.meter import AverageMeter, ProgressMeter
import os


def _move_to_gpu(data, gpu):
    if torch.is_tensor(data) and gpu is not None:
        return data.cuda(gpu, non_blocking=True)
    else:
        return data


class SSLTrainer(BaseTrainer):
    def __init__(self, encoder, projection, loss, optimizer, lr_scheduler, epochs, config,
                 train_data_loader, train_data_sampler=None,
                 evaluator=None, valid_dataloader_sampler=None, gpu=None,
                 writer=None, is_main_trainer=True,

                 save_period=None,
                 save_latest=True,
                 init_val=False,
                 resume_only_encoder=False,

                 scheduler_update='iter',
                 log_step=50,
                 clip_grad=None,
                 mixed_precision=False,
                 ):
        super().__init__(encoder, projection, loss, optimizer, lr_scheduler, epochs, config,
                         save_period=save_period, save_latest=save_latest, init_val=init_val, resume_only_encoder=resume_only_encoder,
                         writer=writer, is_main_trainer=is_main_trainer)
        self.config = config
        self.train_data_loader = train_data_loader
        self.train_data_sampler = train_data_sampler
        if evaluator is None:
            self.evaluators = []
        elif isinstance(evaluator, list):
            self.evaluators = evaluator
        else:
            self.evaluators = [evaluator]
        self.valid_dataloader_sampler = valid_dataloader_sampler
        self.gpu = gpu
        assert scheduler_update in ['iter', 'epoch']
        self.scheduler_update = scheduler_update

        self.log_step = log_step
        self.clip_grad = clip_grad
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        start_time = time.time()

        torch.cuda.empty_cache()
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(self.train_data_loader),
            [data_time, batch_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.encoder.train()
        self.projection.train()

        # apply lr scheduler
        if self.scheduler_update == 'epoch' and self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)

        # reshuffle dataset
        if self.train_data_sampler is not None:
            self.train_data_sampler.set_epoch(epoch)
        else:
            os.environ["WDS_EPOCH"] = str(epoch)

        end = time.time()
        for batch_idx, batch in enumerate(self.train_data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if self.scheduler_update == 'iter' and self.lr_scheduler is not None:
                self.lr_scheduler.step(self.step)

            batch = {key: _move_to_gpu(value, self.gpu) for key, value in batch.items()}

            # compute output and loss
            if self.mixed_precision:
                self.optimizer.zero_grad()

                with autocast():
                    output = self.encoder(batch)
                    output = self.projection(output)

                loss, loss_info = self.loss(batch, output)
                self.scaler.scale(loss).backward()
                losses.update(loss.item())

                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    loss_info['info_grad_norm'] = torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.projection.parameters()), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                output = self.encoder(batch)
                output = self.projection(output)
                loss, loss_info = self.loss(batch, output)
                loss.backward()
                losses.update(loss.item())
                if self.clip_grad is not None:
                    loss_info['info_grad_norm'] = torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.projection.parameters()), self.clip_grad)
                self.optimizer.step()

            # logging
            if self.is_main_trainer:
                if self.writer is not None:
                    self.writer.log_scalar(f'train_loss', losses.val, step=self.step)
                    self.writer.log_scalar(f'data_time', data_time.val, step=self.step)
                    self.writer.log_scalar(f'batch_time', batch_time.val, step=self.step)

                    for loss_name, loss_value in loss_info.items():
                        self.writer.log_scalar(f'train_loss_{loss_name}', loss_value, step=self.step)

                    for i, param_group in enumerate(self.optimizer.param_groups):
                        self.writer.log_scalar(f'lr_param_group_{i}', param_group['lr'], step=self.step)

                if batch_idx % self.log_step == 0:
                    print(progress.display(batch_idx), flush=True)

            self.step += 1
            del batch, output, loss, loss_info

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if self.is_main_trainer and self.writer is not None:
            train_time = time.time() - start_time
            self.writer.log_scalar(f'time_train_epoch', train_time, step=epoch)

        log = {
            f'train_loss': losses.avg,
        }

        return log

    def _valid_epoch(self, epoch):
        logs = {}
        for i, evaluator in enumerate(self.evaluators):
            torch.cuda.empty_cache()
            start_time = time.time()
            val_log = evaluator.validate(
                epoch=epoch,
                encoder=self.encoder,
                projection=self.projection,
                dataloaders_samplers=self.valid_dataloader_sampler,
                gpu=self.gpu,
                logger=self.writer
            )
            if self.writer is not None:
                eval_time = time.time() - start_time
                self.writer.log_scalar(f'time_evaluator_{i}', eval_time, step=epoch)

                for key, value in val_log.items():
                    self.writer.log_scalar(f'{key}', value, step=epoch)
            logs.update(val_log)
        return logs
