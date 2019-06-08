"""Config"""

import argparse

#from action import str2bool

class Config:
    """Config
    """
    def __init__(self):
        """
        parser: to read all config
        config: save config in pairs like key:value
        """
        super(Config, self).__init__()
        self.parser = argparse.ArgumentParser(description='DAC')
        self.config = dict()
        self.args = None
        self._load_necessary()

    def _load_necessary(self):
        """load necessary part in config"""
        self._add_default_setting()
        self._add_special_setting()
        self.args = self.parser.parse_args()
        self._load_default_setting()
        self._load_special_setting()

    def _load_default_setting(self):
        """Load default setting from Parser"""
        self.config['experiment_index'] = self.args.experiment_index
        self.config['cuda'] = self.args.cuda
        self.config["num_workers"] = self.args.num_workers

        self.config['dataset'] = self.args.dataset

        self.config["learning_rate"] = self.args.learning_rate
        self.config['n_epochs'] = self.args.n_epochs
        self.config['batch_size'] = self.args.batch_size

        self.config['seed'] = self.args.seed

        self.config["eval_frequency"] = self.args.eval_frequency
        self.config['log_dir'] = self.args.log_dir
        self.config['model_dir'] = self.args.model_dir

    def _load_special_setting(self):
        self.config["upper_threshold"] = self.args.upper_threshold
        self.config["num_clusters"] = self.args.num_clusters

    def _add_default_setting(self):
        # need defined each time
        self.parser.add_argument('--experiment_index', default="None", type=str,
                                 help="001, 002, ...")
        self.parser.add_argument('--cuda', default='0',
                                 help="cuda visible device")
        self.parser.add_argument("--num_workers", default=2, type=int,
                                 help="num_workers of dataloader")

        self.parser.add_argument('--dataset', default="mnist", type=str,
                                 help="mnist, cifar10, cifar100")

        self.parser.add_argument("--learning_rate", default=1e-3, type=float,
                                 help="learning rate")
        self.parser.add_argument("--batch_size", default=128, type=int,
                                 help="batch size of each epoch")
        self.parser.add_argument("--n_epochs", default=20, type=int,
                                 help="n epochs to train")

        self.parser.add_argument('--seed', default=47, type=int,
                                 help="Random seed for pytorch and Numpy ")

        self.parser.add_argument("--eval_frequency", default=1, type=int,
                                 help="Eval train and test frequency")
        self.parser.add_argument('--log_dir', default="../saved/logdirs/",
                                 type=str, help='store tensorboard files, \
                                 None means not to store')
        self.parser.add_argument('--model_dir', default="../saved/models/",
                                 type=str, help='store models, ../saved/models')

    def _add_special_setting(self):
        self.parser.add_argument("--upper_threshold", default=0.9, type=float,
                                 help="init upper threshold")
        self.parser.add_argument("--num_clusters", default=10, type=int,
                                 help="numbers of clusters")

    def print_config(self):
        """print config
        """
        print('=' * 20, 'basic setting start', '=' * 20)
        for arg in self.config:
            print('{:20}: {}'.format(arg, self.config[arg]))
        print('=' * 20, 'basic setting end', '=' * 20)

    def get_config(self):
        """return config"""
        self.print_config()
        return self.config
