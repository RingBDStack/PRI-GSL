import os
import random
import numpy as np
from collections import Counter
from sklearn.metrics import r2_score


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from .models.graph_clf import GraphClf
from .utils import constants as Constants
from .utils.generic_utils import to_cuda, create_mask
from .utils.constants import INF
from .utils.radam import RAdam


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    def __init__(self, config, train_set=None):
        self.config = config
        self.net_module = GraphClf

        self.criterion = F.nll_loss
        self.score_func = accuracy
        self.metric_name = 'acc'

        # Building network.
        self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('#Parameters = {}\n'.format(num_params))
        self._init_optimizer()


    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['word_embed_dim', 'hidden_size', 'f_qem', 'f_pos', 'f_ner',
                      'word_dropout', 'rnn_dropout',
                      'ctx_graph_hops', 'ctx_graph_topk',
                      'score_unk_threshold', 'score_yes_threshold',
                      'score_no_threshold']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']
        # for k in _ARGUMENTS:
        #     if saved_params['config'][k] != self.config[k]:
        #         print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
        #         self.config[k] = saved_params['config'][k]
        self.graphwave_feature = saved_params['graphwave_feature']
        self.network = self.net_module(self.config)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_new_network(self):
        self.network = self.net_module(self.config)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config.get('lr_reduce_factor', 0.5), \
                    patience=self.config.get('lr_patience', 200), verbose=True)

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'graphwave_feature': self.graphwave_feature,
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')


    def clip_grad(self):
        # Clip gradients
        if self.config.get('grad_clipping', False):
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)
