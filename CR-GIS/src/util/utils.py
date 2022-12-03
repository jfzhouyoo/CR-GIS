import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import torch.nn as nn
import math
import argparse

from pytz import timezone, utc
from datetime import datetime

def shanghai_format(sec, what):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("Asia/Shanghai")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


def get_logger(logname):
    logging.Formatter.converter = shanghai_format
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s  [%(levelname)s]  %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Pack(dict):
    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        pack_list = []
        for vs in zip(*self.values()):
            pack = Pack(zip(self.keys(), vs))
            pack_list.append(pack)
        return pack_list

    def cuda(self, device=None):
        pack = Pack()
        for k, v in self.items():
            if "oovs_str" not in k:
                if isinstance(v, tuple):
                    pack[k] = tuple(x.cuda(device) for x in v)
                else:
                    pack[k] = v.cuda(device)
            else:
                pack[k] = v
        return pack


def max_lens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor
    
    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i,:l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i,j,:l] = torch.tensor(x)
                lengths[i,j] = l
    return tensor, lengths


def _create_entity_embeddings(entity_num, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(entity_num, embedding_size)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        assert 0 not in actual[i] # 0 is padding idx
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    true_users = 0
    for user_id in range(len(actual)):
        assert 0 not in actual[user_id] # 0 is padding idx
        if len(actual) == 0:
            continue
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
        true_users += 1
    # return res / float(len(actual))
    return res / float(true_users) # only computer recommendation sample

def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def str2bool(v):
    """
    str2bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')