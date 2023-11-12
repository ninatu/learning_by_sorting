import os
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from learning_by_sorting.utils.utils import write_yaml, read_config
import time
import inspect



class ConfigParser:
    def __init__(self, args, options='', timestamp=True, eval=False, parse_from_string=None):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        if parse_from_string is not None:
            import shlex
            args = args.parse_args(shlex.split(parse_from_string))
        else:
            args = args.parse_args()
        self.args = args

        if args.resume is None:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.cfg_fname = Path(args.config)
            config = read_config(self.cfg_fname)
            self.resume = None
        else:
            self.resume = Path(args.resume)
            resume_cfg_fname = self.resume.parent / 'config.yaml'
            config = read_config(resume_cfg_fname)
            if args.config is not None:
                config.update(read_config(Path(args.config)))

        # load config file and apply custom cli options
        self.config = _update_config(config, options, args)

        if args.name is not None:
            config['name'] = args.name
        elif args.config is not None:
            config['name'] = os.path.splitext(os.path.basename(args.config))[0]

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['save_dir'])
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''

        exper_name = self.config['name']
        self._models_dir = save_dir / 'models' / exper_name / timestamp

        if not eval:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        # if set, remove all previous experiments with the current config
        if vars(args).get("purge_exp_dir", False):
            for dirpath in (self._models_dir,):
                config_dir = dirpath.parent
                existing = list(config_dir.glob("*"))
                print(f"purging {len(existing)} directories from config_dir...")
                tic = time.time()
                os.system(f"rm -rf {config_dir}")
                print(f"Finished purge in {time.time() - tic:.3f}s")

        # save updated config file to the checkpoint dir
        if not eval:
            write_yaml(self.config, self.model_dir / 'config.yaml')

    def initialize(self, name, module,  *args, index=None, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the 
        instance initialized with corresponding keyword args given as 'args'.
        """
        if index is None:
            module_name = self[name]['type']
            module_args = dict(self[name]['args'])
        else:
            module_name = self[name][index]['type']
            module_args = dict(self[name][index]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)

        # if parameter not in config subdict, then check if it's in global config.
        signature = inspect.signature(getattr(module, module_name).__init__)
        for param in signature.parameters.keys():
            if param not in module_args and param in self.config:
                module_args[param] = self[param]

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    @property
    def model_dir(self):
        return self._models_dir


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
