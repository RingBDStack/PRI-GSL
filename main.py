import argparse
import yaml
import torch
import pprint
import warnings
import numpy as np
import random
from collections import defaultdict, OrderedDict
import os, sys
sys.path.append(os.path.dirname(__file__))
from core.model_handler import ModelHandler

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(config, dataset=None):
    # print_config(config)
    set_random_seed(config['random_seed'])
    if config.get('multi_run', False):
       max_run = 10
    else:
        max_run = 1
    test_acc = []
    for run in range(max_run):
        config['fold'] = run
        set_random_seed(np.random.randint(50))
        model = ModelHandler(config, dataset)
        model.train(multi_run=run)
        test_metrics = model.test()
        test_acc.append(test_metrics['acc'])
    return np.mean(test_acc)

def test_performance(config, dataset=None):
    set_random_seed(config['random_seed'])
    # set_random_seed(np.random.randint(50))
    model = ModelHandler(config, dataset)
    model.dirname = config['dirname']
    test_metrics = model.test()
    return test_metrics['acc']

################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="actor", help="Folder with training graph jsons.")
    parser.add_argument("--use_config", type=int, default=0, help="Use default config file to override the command-line.")
    parser.add_argument("--config_path", type=str, default="./config/", help="The default config file path.")
    parser.add_argument("--dirname", type=str, default=None, help="Saved param file path.")
    parser.add_argument("--fold", type=int, default=0, help="Default is 10.")
    parser.add_argument("--backbone", type=str, default="GCN", help="GCN, GAT, GIN")
    parser.add_argument("--graph_type", type=str, default="prob", choices=['epsiloneNN', 'KNN', 'prob'], help="epsilonNN, KNN, prob")
    parser.add_argument("--graph_metric_type", type=str, default="weighted_cosine")
    parser.add_argument("--num_layers", type=int, default=2, help="Default is 2.")
    parser.add_argument("--hidden_size", type=int, default=16, help="Default is 32.")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Number of epochs. Default is 2000.")
    parser.add_argument("--patience", type=int, default=300, help="Stop when have no improvement for nearly 300 epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate. Default is 0.01.")
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=5*10**-5, help="Adam weight decay. Default is 5*10^-5.")
    # parser.add_argument("--batch_size", type=int, default=10, help="batch_size")

    # config of pri
    parser.add_argument("--use_pri", type=int, default=1, help="whether to use pri")
    parser.add_argument("--pri_loss_weight", type=float, default=0.1, help="the weight of pri loss, default is 0.1")
    parser.add_argument("--beta", type=float, default=0.5, help="")

    # config of graphwave
    parser.add_argument("--use_wave", type=int, default=1, help="whether to use graphwave")
    parser.add_argument("--wave_weight", type=float, default=0.01, help="the weight of wave feature")
    parser.add_argument("--wave_learn_shape", type=int, default=16, help="the learning weight shape in construct learning")
    parser.add_argument("--update_feature", type=int, default=200, help="calculate wave feature when nearly 200 epochs")

    # config of IDGL regularization, defalut is all False (0.0)
    parser.add_argument("--smoothness_ratio", type=float, default=0.0, help="")
    parser.add_argument("--sparsity_ratio", type=float, default=0.0, help="")
    parser.add_argument("--graph_learn_regularization", type=bool, default=False, help="")

    parser.add_argument("--degree_ratio", type=float, default=0.8, help="")
    parser.add_argument("--graph_learn_ratio", type=float, default=0.5, help="")
    parser.add_argument("--graph_learn_epsilon", type=float, default=0.3, help="")

    # config of 
    parser.add_argument("--eps_adj", type=float, default=4e-5, help="")
    parser.add_argument("--data_seed", type=int, default=42, help="")
    parser.add_argument("--verbose", type=int, default=200, help="")
    parser.add_argument("--print_every_epochs", type=int, default=10, help="")
    parser.add_argument("--graph_learn", type=float, default=0.01, help="")
    parser.add_argument("--graph_learn_num_pers", type=int, default=4, help="")
    parser.add_argument("--random_seed", type=int, default=1234, help="")
    parser.add_argument("--early_stop_metric", type=str, default='acc', help="")
    parser.add_argument("--update_adj_ratio", type=float, default=0.1, help="")

    # basic config
    parser.add_argument("--data_dir", type=str, default="./data/", help="")
    parser.add_argument("--dropout", type=float, default=0.5, help="")

    args = vars(parser.parse_args())
    return args

################################################################################
# Module Command-line Behavior #
################################################################################
def get_dir_path(cur_dir_path):
    return os.path.abspath(cur_dir_path)

if __name__ == '__main__':

    cfg = get_args()
    if cfg['use_config']:
        try:
            if cfg['dataset_name'] in ['cora', 'citeseer', 'squirrel', 'actor', 'chameleon', 'photo']:
                cfg['config'] = f"{cfg['config_path']}/{cfg['dataset_name']}.yml"
            config = get_config(cfg['config'])
            print(f"We use {cfg['config']} to override the command-line parameters.")
        except FileNotFoundError:
            warnings.warn(f"We cannot find the config file in {cfg['config_path']}, please check the path \
                or setting --use_config to False. We use command-line parameters to continue.")
            config = {}
        cfg.update({k:v for k,v in config.items()})
    pprint.pprint(cfg)
    if cfg['dirname'] is not None:
        test_performance(cfg)
    else:
        cfg['dirname'] = os.path.join(os.getcwd(), 'save_dirs', cfg['dataset_name'])
        test_acc = main(cfg)
        print(f"save to {cfg['dirname']}")
        
        
