import os
import time
import json
import glob
import numpy as np
import tqdm
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from .model import Model
from .utils.generic_utils import to_cuda
# from .utils.data_utils import prepare_datasets, DataStream, vectorize_input

from .utils.prepare_dataset import prepare_datasets
from .utils import Timer, DummyLogger, AverageMeter
from .utils import constants as Constants
from .layers.common import dropout
from .layers.anchor import sample_anchors, batch_sample_anchors, batch_select_from_tensor, compute_anchor_adj
from .models.PRILoss import pri_node
from .models.graphwave import graphwave_alg

class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config, datasets):
        # Evaluation Metrics:
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self._train_metrics = {'nloss': AverageMeter(),
                            'acc': AverageMeter()}
        self._dev_metrics = {'nloss': AverageMeter(),
                            'acc': AverageMeter()}

        self.device = torch.device("cuda")
        config['device'] = self.device


        if datasets is None:
            datasets, num_nodes = prepare_datasets(config, fold=config['fold'])


        # Prepare datasets
        config['num_feat'] = datasets['features'].shape[-1]
        config['num_class'] = datasets['labels'].max().item() + 1

        # Initialize the model
        self.model = Model(config, train_set=datasets.get('train', None))
        self.model.network = self.model.network.to(self.device)


        self._n_test_examples = datasets['idx_test'].shape[0]
        self.run_epoch = self._scalable_run_whole_epoch if config.get('scalable_run', False) else self._run_whole_epoch

        self.train_loader = datasets
        self.dev_loader = datasets
        self.test_loader = datasets

        self.config = self.model.config
        self.is_test = False


    def train(self, multi_run):
        if self.train_loader is None or self.dev_loader is None:
            print("No training set or dev set specified -- skipped training.")
            return

        self.is_test = False
        timer = Timer("Train")
        self._epoch = self._best_epoch = 0

        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        self._reset_metrics()

        # for self._epoch in tqdm.trange(self.config['max_epochs']):
        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1
            if not self._stop_condition(self._epoch, self.config['patience']):
                break

            # Train phase
            # if self._epoch % self.config['print_every_epochs'] == 0:
            #     format_str = "\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs'])
            #     print(format_str)
            #     self.logger.write_to_file(format_str)

            self.run_epoch(self.train_loader, training=True, verbose=self.config['verbose'])

            # Validation phase
            dev_output, dev_gold = self.run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'],
                                 out_predictions=self.config.get('out_predictions', True))
            test_output, test_gold = self.run_epoch(self.test_loader, training=False, verbose=self.config['verbose'],
                                 out_predictions=self.config.get('out_predictions', True))
            if self.config.get('out_predictions', True):
                dev_metric_score = self.model.score_func(dev_gold, dev_output)
            else:
                dev_metric_score = None
            val_score = self.model.score_func(dev_gold, dev_output)
            test_score = self.model.score_func(test_gold, test_output)

            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Validation Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._dev_loss.mean())
                format_str += self.metric_to_str(self._dev_metrics)
                if dev_metric_score is not None:
                    format_str += '\n Dev score: {:0.5f}'.format(dev_metric_score)
                dev_epoch_time_msg = timer.interval("Validation Epoch {}".format(self._epoch))
                # self.logger.write_to_file(dev_epoch_time_msg + '\n' + format_str)
                print(format_str)

            # self.model.scheduler.step(self._dev_metrics[self.config['eraly_stop_metric']].mean())

            if self.config['early_stop_metric'] == self.model.metric_name and dev_metric_score is not None:
                cur_dev_score = dev_metric_score
            else:
                cur_dev_score = self._dev_metrics[self.config['early_stop_metric']].mean()

            # if self._best_metrics[self.config['eraly_stop_metric']] < self._dev_metrics[self.config['eraly_stop_metric']].mean():
            if self._best_metrics[self.config['early_stop_metric']] < cur_dev_score:
                self._best_epoch = self._epoch
                if not os.path.exists(self.config['dirname']):
                    os.makedirs(self.config['dirname'])
                self.model.save(self.config['dirname'])
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()

                if dev_metric_score is not None:
                    self._best_metrics[self.model.metric_name] = dev_metric_score

                    if self._epoch % self.config['print_every_epochs'] == 0:
                        print(format_str)

                if self._epoch % self.config['print_every_epochs'] == 0:
                    format_str = "!!! Updated: " + self.best_metric_to_str(self._best_metrics)
                    # self.logger.write_to_file(format_str)
                    print(format_str)

            self._reset_metrics()

        timer.finish()

        # format_str = "Finished Training: {}\nTraining time: {}".format(self.dirname, timer.total) + '\n' + self.summary()
        format_str = self.summary()
        print(format_str)
        # self.logger.write_to_file(format_str)
        return self._best_metrics


    def test(self):
        self._epoch = 1
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return

        # Restore best model
        print('Restoring best model')
        self.model.init_saved_network(self.config['dirname'])
        self.model.network = self.model.network.to(self.device)
        self.model.graphwave_feature = self.model.graphwave_feature.to(self.device)

        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")
        for param in self.model.network.parameters():
            param.requires_grad = False

        output, gold = self.run_epoch(self.test_loader, training=False, verbose=0,
                                 out_predictions=self.config.get('out_predictions', True))

        metrics = self._dev_metrics
        format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
            self._n_test_examples, 1, 1)
        format_str += self.metric_to_str(metrics)

        if self.config.get('out_predictions', True):
            test_score = self.model.score_func(gold, output)
            format_str += '\nFinal score on the testing set: {:0.5f}\n'.format(test_score)
        # else:
        #     test_score = None

        # print(format_str)
        # self.logger.write_to_file(format_str)
        # timer.finish()

        # format_str = "Finished Testing: {}\nTesting time: {}".format(self.dirname, timer.total)
        print(format_str)
        # self.logger.write_to_file(format_str)
        # self.logger.close()

        test_metrics = {}
        for k in metrics:
            test_metrics[k] = metrics[k].mean()

        if test_score is not None:
            test_metrics[self.model.metric_name] = test_score
        return test_metrics

    def _run_whole_epoch(self, data_loader, training=True, verbose=None, out_predictions=False):
        '''BP after all iterations'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.network.train(training)

        init_adj, features, labels = data_loader['adj'], data_loader['features'].to(torch.float32), data_loader['labels']

        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']
        else:
            idx = data_loader['idx_test']

        network = self.model.network


        # Init
        features = F.dropout(features, network.config.get('feat_adj_dropout', 0), training=network.training)
        init_node_vec = features

        # add_feature_0
        # if self.config.get("use_wave", True):
        #     graphwave_feature, _, _ = graphwave_alg(init_adj, np.linspace(0, 100, self.config['wave_init_shape']//4), taus='auto')
        #     add_wave_feature = torch.cat([init_node_vec, self.config.get("wave_weight", 0.01) * graphwave_feature.to(init_node_vec.device)], dim=1)
        # else:
        #     add_wave_feature = init_node_vec
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, init_node_vec, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)
        if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
            cur_raw_adj = F.dropout(cur_raw_adj, network.config.get('feat_adj_dropout', 0), training=network.training)
        # cur_adj = F.dropout(cur_adj, network.config.get('feat_adj_dropout', 0), training=network.training)


        if network.graph_module == 'gat':
            assert self.config['graph_learn'] is False and self.config.get('max_iter', 10) == 0
            node_vec = network.encoder(init_node_vec, init_adj)
            output = F.log_softmax(node_vec, dim=-1)

        elif network.graph_module == 'graphsage':
            assert self.config['graph_learn'] is False and self.config.get('max_iter', 10) == 0
            # Convert adj to DGLh
            import dgl
            from scipy import sparse
            binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj)

            node_vec = network.encoder(dgl_graph, init_node_vec)
            output = F.log_softmax(node_vec, dim=-1)

        else:
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, init_adj)
            output = F.log_softmax(output, dim=-1)


        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        # if self.config['graph_learn'] and self.config['graph_learn_regularization']:
        #     loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)
        first_raw_adj, first_adj = cur_raw_adj, cur_adj
        first_adj_copy = torch.zeros_like(cur_adj).copy_(cur_adj)

        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)
        
        if  training and self.config.get('use_wave', True) and (self._epoch - 1)  % self.config['update_feature'] == 0:
            graphwave_feature, _, _ = graphwave_alg(cur_adj.clone().detach(), np.linspace(0, 100, self.config['wave_learn_shape']//4), taus='auto')
            self.model.graphwave_feature = torch.from_numpy(graphwave_feature).float().to(node_vec.device)
            del graphwave_feature
            print(f"epoch {self._epoch}: updateing feature.")

        if training:
            eps_adj = float(self.config.get('eps_adj', 0))
        else:
            eps_adj = float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))

        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj

        loss = 0
        iter_ = 0
        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < max_iter_:
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            # add_feature_1
            if self.config.get('use_wave', True):
                add_wave_feature = torch.cat([node_vec, self.config.get("wave_weight", 0.0) * self.model.graphwave_feature], dim=1)
            else:
                add_wave_feature = node_vec
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, add_wave_feature, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)


            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj

            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)
            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])
            # val_score = self.model.score_func(labels[data_loader['idx_val']], output[data_loader['idx_val']])
            # print(f"val_score:{val_score}")

            # print("loss:", loss.item(), "score:", score, "test_acc:", self.model.score_func(labels[data_loader['idx_val']], output[data_loader['idx_val']]))

            # PRI
            if self.config['graph_learn'] and self.config.get("use_pri", True):
                loss += self.config.get('pri_loss_weight', 0.3)*pri_node(cur_raw_adj, init_adj, None, None, beta=self.config.get('beta'))

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += self.add_graph_loss(cur_raw_adj, init_node_vec)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')

        # if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
        #     out_raw_learned_adj_path = os.path.join(self.dirname, self.config['out_raw_learned_adj_path'])
        #     np.save(out_raw_learned_adj_path, cur_raw_adj.cpu())
        #     print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        if training:
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grad()
            self.model.optimizer.step()

        self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        return output[idx], labels[idx]


    def _scalable_run_whole_epoch(self, data_loader, training=True, verbose=None, out_predictions=False):
        '''Scalable run: BP after all iterations'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.network.train(training)

        init_adj, features, labels = data_loader['adj'], data_loader['features'], data_loader['labels']

        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']
        else:
            idx = data_loader['idx_test']

        network = self.model.network

        # Init
        features = F.dropout(features, network.config.get('feat_adj_dropout', 0), training=network.training)
        init_node_vec = features

        # Randomly sample s anchor nodes
        init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, network.config.get('num_anchors', int(0.2 * init_node_vec.size(0))))

        # Compute n x s node-anchor relationship matrix
        graphwave_feature, _, _ = graphwave_alg(init_adj, np.linspace(0, 100, 250), taus='auto')
        add_wave_feature = torch.cat([init_anchor_vec, self.config.get("wave_weight", 0.01) * graphwave_feature.to(init_node_vec.device)[sampled_node_idx]], dim=1)
        init_node_vec = torch.cat([init_node_vec, self.config.get("wave_weight", 0.01) * graphwave_feature.to(init_node_vec.device)], dim=1)
        # cur_node_anchor_adj = network.learn_graph(network.graph_learner, init_anchor_vec, anchor_features=init_anchor_vec)
        cur_node_anchor_adj = network.learn_graph(network.graph_learner, init_node_vec, anchor_features=add_wave_feature)

        # Compute s x s anchor graph
        cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)


        if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
            cur_node_anchor_adj = F.dropout(cur_node_anchor_adj, network.config.get('feat_adj_dropout', 0), training=network.training)

        cur_anchor_adj = F.dropout(cur_anchor_adj, network.config.get('feat_adj_dropout', 0), training=network.training)

        # Update node embeddings via node-anchor-node message passing
        init_agg_vec = network.encoder.graph_encoders[0](init_node_vec, init_adj, anchor_mp=False, batch_norm=False)
        node_vec = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                    network.graph_skip_conn * init_agg_vec

        if network.encoder.graph_encoders[0].bn is not None:
            node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

        node_vec = torch.relu(node_vec)
        node_vec = F.dropout(node_vec, network.dropout, training=network.training)
        anchor_vec = node_vec[sampled_node_idx]


        first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
        first_init_agg_vec = network.encoder.graph_encoders[0](init_node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)


        # Add mid GNN layers
        for encoder in network.encoder.graph_encoders[1:-1]:
            node_vec = (1 - network.graph_skip_conn) * encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                        network.graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

            if encoder.bn is not None:
                node_vec = encoder.compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)
            anchor_vec = node_vec[sampled_node_idx]


        # Compute output via node-anchor-node message passing
        output = (1 - network.graph_skip_conn) * network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                    network.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj, anchor_mp=False, batch_norm=False)
        output = F.log_softmax(output, dim=-1)
        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_graph_loss(cur_anchor_adj, init_anchor_vec)


        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)


        if training:
            eps_adj = float(self.config.get('eps_adj', 0)) # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
        else:
            eps_adj = float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))


        pre_node_anchor_adj = cur_node_anchor_adj

        loss = 0
        iter_ = 0
        # graphwave_feature, _, _ = graphwave_alg(init_adj.clone().detach(), np.linspace(0, 100, 4), taus='auto')
        # anchor_vec = torch.cat([anchor_vec, self.config.get("wave_weight", 0.01) * graphwave_feature.to(init_node_vec.device)[sampled_node_idx]], dim=1)
        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj).item() > eps_adj) and iter_ < max_iter_:
            iter_ += 1
            pre_node_anchor_adj = cur_node_anchor_adj

            # Compute n x s node-anchor relationship matrix
            # graphwave_feature, _, _ = graphwave_alg(cur_adj.clone().detach(), np.linspace(0, 100, 4), taus='auto')
            # add_wave_feature = torch.cat([node_vec, self.config.get("wave_weight", 0.0) * graphwave_feature.to(init_node_vec.device).detach()], dim=1)
            anchor_vec = torch.cat([anchor_vec, self.config.get("wave_weight", 0.01) * graphwave_feature.to(init_node_vec.device)[sampled_node_idx]], dim=1)
            node_vec = torch.cat([node_vec, self.config.get("wave_weight", 0.01) * graphwave_feature.to(init_node_vec.device)], dim=1)
            cur_node_anchor_adj = network.learn_graph(network.graph_learner2, node_vec, anchor_features=anchor_vec)

            # Compute s x s anchor graph
            cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

            cur_agg_vec = network.encoder.graph_encoders[0](init_node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (1 - update_adj_ratio) * first_init_agg_vec

            node_vec = (1 - network.graph_skip_conn) * cur_agg_vec + \
                    network.graph_skip_conn * init_agg_vec

            if network.encoder.graph_encoders[0].bn is not None:
                node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)
            anchor_vec = node_vec[sampled_node_idx]


            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                mid_cur_agg_vec = encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
                if update_adj_ratio is not None:
                    mid_first_agg_vecc = encoder(node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
                    mid_cur_agg_vec = update_adj_ratio * mid_cur_agg_vec + (1 - update_adj_ratio) * mid_first_agg_vecc

                node_vec = (1 - network.graph_skip_conn) * mid_cur_agg_vec + \
                        network.graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)

                if encoder.bn is not None:
                    node_vec = encoder.compute_bn(node_vec)

                node_vec = torch.relu(node_vec)
                node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)
                anchor_vec = node_vec[sampled_node_idx]


            cur_agg_vec = network.encoder.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
            if update_adj_ratio is not None:
                first_agg_vec = network.encoder.graph_encoders[-1](node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (1 - update_adj_ratio) * first_agg_vec

            output = (1 - network.graph_skip_conn) * cur_agg_vec + \
                    network.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj, anchor_mp=False, batch_norm=False)

            output = F.log_softmax(output, dim=-1)
            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += self.add_graph_loss(cur_anchor_adj, init_anchor_vec)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_node_anchor_adj - pre_node_anchor_adj) * self.config.get('graph_learn_ratio')

        if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
            out_raw_learned_adj_path = os.path.join(self.dirname, self.config['out_raw_learned_adj_path'])
            np.save(out_raw_learned_adj_path, cur_node_anchor_adj.cpu())
            print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1
        if training:
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grad()
            self.model.optimizer.step()

        self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        return output[idx], labels[idx]

    def _run_batch_epoch(self, data_loader, training=True, rl_ratio=0, verbose=10, out_predictions=False):
        start_time = time.time()
        mode = "train" if training else ("test" if self.is_test else "dev")

        if training:
            self.model.optimizer.zero_grad()
        output = []
        gold = []
        for step in range(data_loader.get_num_batch()):
            input_batch = data_loader.nextBatch()
            x_batch = vectorize_input(input_batch, self.config, training=training, device=self.device)
            if not x_batch:
                continue  # When there are no examples in the batch

            if self.config.get('no_gnn', False):
                res = self.batch_no_gnn(x_batch, step, training=training, out_predictions=out_predictions)
            else:
                if self.config.get('scalable_run', False):
                    res = self.scalable_batch_IGL_stop(x_batch, step, training=training, out_predictions=out_predictions)
                else:
                    res = self.batch_IGL_stop(x_batch, step, training=training, out_predictions=out_predictions)

            loss = res['loss']
            metrics = res['metrics']
            self._update_metrics(loss, metrics, x_batch['batch_size'], training=training)

            if training:
                self._n_train_examples += x_batch['batch_size']

            if (verbose > 0) and (step > 0) and (step % verbose == 0):
                summary_str = self.self_report(step, mode)
                self.logger.write_to_file(summary_str)
                print(summary_str)
                print('used_time: {:0.2f}s'.format(time.time() - start_time))

            if not training and out_predictions:
                output.extend(res['predictions'])
                gold.extend(x_batch['targets'])
        return output, gold


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "dev":
            format_str = "[predict-{}] step: [{} / {}] | loss = {:0.5f}".format(
                    self._epoch, step, self._n_dev_batches, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
        elif mode == "test":
            format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
                    self._n_test_examples, step, self._n_test_batches)
            format_str += self.metric_to_str(self._dev_metrics)
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def plain_metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True


    def add_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.config['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
        graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss


    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(self.config['smoothness_ratio'] * torch.trace(torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape[1:])))

            graph_loss = to_cuda(torch.Tensor(graph_loss), self.device)

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze(-1).squeeze(-1) / out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2), (1, 2)) / int(np.prod(out_adj.shape[1:]))

        else:
            graph_loss = 0
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features[i].transpose(-1, -2), torch.mm(L, features[i]))) / int(np.prod(out_adj.shape))

            ones_vec = to_cuda(torch.ones(out_adj.shape[:-1]), self.device)
            graph_loss += -self.config['degree_ratio'] * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).sum() / out_adj.shape[0] / out_adj.shape[-1]
            graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_


def batch_diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2)) # Shape: [batch_size]
    norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_

def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))

def batch_SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))

