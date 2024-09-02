import os
import sys
import argparse
from os.path import join as ospj


_DATASET = ('UCMerced', 'MLRSNet', 'AID')
_SCHEMES = ('BCE', 'ELR', 'SAT', 'LCR', 'JoCoR', 'RCML')
_LOOKUP = {
    'num_classes': {
        'UCMerced': 17,
        'MLRSNet': 60,
        'AID': 17,
    },
    'num_train': {
        'UCMerced': 1680,
        'MLRSNet': 87319,
        'AID': 2400,
    },
    'image_size': {
        'UCMerced': 256,
        'MLRSNet': 256,
        'AID': 600,
    },
    'path_to_dataset': {
        'UCMerced': 'data/UCMerced_LandUse',
        'MLRSNet': 'data/MLRSNet',
        'AID': 'data/AID_multilabel',
    }
}

class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self.log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def set_dir(runs_dir, exp_name):
    runs_dir = ospj(runs_dir, exp_name)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    return runs_dir

def set_log(runs_dir):
    log_file_name = ospj(runs_dir, 'log.log')
    Logger(log_file_name)

def set_follow_up_configs(args):
    args.feat_dim = 2048
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    args.num_train = _LOOKUP['num_train'][args.dataset]
    args.image_size = _LOOKUP['image_size'][args.dataset]
    args.path_to_dataset = _LOOKUP['path_to_dataset'][args.dataset]
    args.save_path = set_dir(args.save_path, args.experiment_name)

    return args


def get_configs():
    parser = argparse.ArgumentParser()

    # Default settings
    parser.add_argument('--num_epochs', type=int, default=50)
    
    # Util
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--experiment_name', type=str, default='exp_default')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_num', type=int, default=0)
    
    # Data
    parser.add_argument('--dataset', type=str, required=True,
                        choices=_DATASET)

    # Hyperparameters
    parser.add_argument('--bsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_mult', type=float, default=10)
    parser.add_argument('--scheme', type=str, required=True, 
                        choices=_SCHEMES)
    parser.add_argument('--lam1', type=float, default=1)
    parser.add_argument('--lam2', type=float, default=0.1)
    parser.add_argument('--lam3', type=float, default=0.6)
    parser.add_argument('--Es', type=int, default=5)
    parser.add_argument('--Tk', type=int, default=50)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--swap_rate', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.6)

    parser.add_argument('--noise_rate', type=float, default=0)

    args = parser.parse_args()
    args = set_follow_up_configs(args)
    set_log(args.save_path)

    return args


