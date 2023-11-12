import torch
from abc import abstractmethod
import os
from collections import OrderedDict


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, encoder, projection, loss, optimizer, lr_scheduler, epochs, config,
                 save_period=None, save_latest=True, init_val=False,  resume_only_encoder=False,
                 writer=None, is_main_trainer=True):
        self.config = config
        self.encoder = encoder
        self.projection = projection
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_main_trainer = is_main_trainer

        self.epochs = epochs
        self.save_period = save_period
        self.save_latest = save_latest
        self.init_val = init_val

        self.start_epoch = 0
        self.step = 0
        self.checkpoint_dir = config.model_dir

        self.writer = writer
        if config.resume is not None:
            self._resume_checkpoint(config.resume, resume_only_encoder=resume_only_encoder)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        if self.init_val:
            log = self._valid_epoch(self.start_epoch - 1)
            if self.is_main_trainer:
                for key, value in log.items():
                    print('    {:15s}: {}'.format(str(key), value))

        for epoch in range(self.start_epoch, self.epochs):
            log = self._train_epoch(epoch)

            if self.is_main_trainer:
                if self.save_latest:
                    self._save_checkpoint(epoch, save_latest=True)

                if (self.save_period is not None) and ((epoch + 1) % self.save_period == 0):
                    self._save_checkpoint(epoch)

            val_log = self._valid_epoch(epoch)
            log.update(val_log)

            if self.is_main_trainer:
                log['epoch'] = epoch
                for key, value in log.items():
                    print('    {:15s}: {}'.format(str(key), value))

    def _save_checkpoint(self, epoch, save_latest=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        state = {
            'epoch': epoch,
            'step': self.step,
            'encoder': self.encoder.state_dict(),
            'projection': self.projection.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config.config
        }
        if save_latest:
            path = str(self.checkpoint_dir / 'latest_model.pth')
        else:
            path = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')

        print("Saving checkpoint: {} ...".format(path))
        # for safety (in case of the "out of memory" error in order to not lose the latest checkpoint)
        tmp_path = str(self.checkpoint_dir / 'tmp.pth')
        torch.save(state, tmp_path, _use_new_zipfile_serialization=False)
        os.rename(tmp_path, path)

    def _resume_checkpoint(self, resume_path, resume_only_encoder=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path, map_location='cpu')

        print("Checkpoint is from epoch {}".format(checkpoint['epoch']))

        encoder_state_dict = fix_module_in_state_dict(checkpoint['encoder'], self.encoder)
        self.encoder.load_state_dict(encoder_state_dict)

        if not resume_only_encoder:
            self.start_epoch = checkpoint['epoch'] + 1

            if 'step' in checkpoint:
                self.step = checkpoint['step'] + 1

            projection_state_dict = fix_module_in_state_dict(checkpoint['projection'], self.projection)
            self.projection.load_state_dict(projection_state_dict)
            del projection_state_dict

            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded.")
        del checkpoint, encoder_state_dict
        torch.cuda.empty_cache()


def fix_module_in_state_dict(state_dict, model):
    load_state_dict_keys = list(state_dict.keys())
    curr_state_dict_keys = list(model.state_dict().keys())
    redo_dp = False
    if not curr_state_dict_keys[0].startswith('module.') and load_state_dict_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_state_dict_keys[0].startswith('module.') and not load_state_dict_keys[0].startswith('module.'):
        redo_dp = True
        undo_dp = False
    else:
        undo_dp = False

    if undo_dp:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    return new_state_dict
