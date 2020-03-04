from dataloader import EvalDataset, TrainDataset
from dataloader import get_dataset

import argparse
import os
import logging
import time
import pickle
import numpy as np
import scipy as sp
import torch as th

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    import multiprocessing as mp
    from train_mxnet import load_model_from_checkpoint
    from train_mxnet import test
else:
    import torch.multiprocessing as mp
    from train_pytorch import load_model_from_checkpoint
    from train_pytorch import test, test_mp
    from train_pytorch import infer

import dgl.backend as F
import dgl

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransH', 'TransR', 'TransD',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE'],
                          help='model to use')
        self.add_argument('--data_path', type=str, default='data',
                          help='root path of all dataset')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='dataset name, under data_path')
        self.add_argument('--format', type=str, default='built_in',
                          choices=['built_in', 'raw_udd', 'udd'],
                          help='the format of the dataset.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='a list of data files, e.g. entity relation train valid test')
        self.add_argument('--predict_head', type=str, default=None)
        self.add_argument('--predict_tail', type=str, default=None)
        self.add_argument('--model_path', type=str, default='ckpts',
                          help='the place where models are saved')

        self.add_argument('--batch_size', type=int, default=8,
                          help='batch size used for eval and test')
        self.add_argument('--hidden_dim', type=int, default=256,
                          help='hidden dim used by relation and entity')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='margin value')

        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='a list of active gpu ids, e.g. 0')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='mix CPU and GPU training')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='double entitiy dim for complex number')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='double relation dim for complex number')

    def parse_args(self):
        args = super().parse_args()
        return args

def get_logger(args):
    if not os.path.exists(args.model_path):
        raise Exception('No existing model_path: ' + args.model_path)

    log_file = os.path.join(args.model_path, 'eval.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    logger = logging.getLogger(__name__)
    print("Logs are being recorded at: {}".format(log_file))
    return logger

def main(args):
    # load dataset and samplers
    dataset = get_dataset(args.data_path, args.dataset, args.format, args.data_files)
    args.batch_size_eval = args.batch_size

    args.pickle_graph = False
    args.train = False
    args.valid = False
    args.test = True
    args.strict_rel_part = False
    args.soft_rel_part = False
    args.async_update = False
    logger = get_logger(args)
    # Here we want to use the regualr negative sampler because we need to ensure that
    # all positive edges are excluded.
    src = np.concatenate((dataset.train[0], dataset.valid[0], dataset.test[0]))
    etype_id = np.concatenate((dataset.train[1], dataset.valid[1], dataset.test[1]))
    dst = np.concatenate((dataset.train[2], dataset.valid[2], dataset.test[2]))
    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                shape=[dataset.n_entities, dataset.n_entities])
    g = dgl.DGLGraph(coo, readonly=True, sort_csr=True)
    g.edata['tid'] = F.tensor(etype_id, F.int64)

    # load model
    n_entities = dataset.n_entities
    n_relations = dataset.n_relations
    ckpt_path = args.model_path
    model = load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path)

    head_name = []
    tail_name = []
    head = []
    tail = []
    relation = [5]
    with open(args.predict_head) as f:
        for line in f:
            name, h = line.strip().split('\t')
            head.append(int(h))
            head_name.append(name)

    with open(args.predict_tail) as f:
        for line in f:
            name, t = line.strip().split('\t')
            tail.append(int(t))
            tail_name.append(name)

    head = F.tensor(np.array(head), F.int64)
    relation = F.tensor(np.array(relation), F.int64)
    tail = F.tensor(np.array(tail), F.int64)
    
    scores = infer(args, model, (head, relation, tail), 'topk_head')
    top10 = []
    for idx, score in enumerate(scores[0]):
        sort, indices = th.sort(score)
        scores = sort[-10:]
        indices = indices[-10:]
        #heads = head_name[indices]
        head_r = "{}".format(head_name[indices[0]])
        for i in range(9):
            head_r = "{}\t{}".format(head_r, head_name[indices[i+1]])
        score_r = "{}\t".format(scores[0])
        for i in range(9):
            score_r = "{}\t{}".format(score_r, scores[i+1])
        result = "{}\t{}\t{}\n".format(tail_name[idx], head_r, score_r)
        print(result)
        top10.append(result)

    with open("topk.tsv", "w+") as f:
        f.writelines(top10)

if __name__ == '__main__':
    args = ArgParser().parse_args()
    main(args)

