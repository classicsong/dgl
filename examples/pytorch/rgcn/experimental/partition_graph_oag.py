import os
import dgl
import numpy as np
import torch as th
import argparse
import time
from sklearn.preprocessing import MultiLabelBinarizer

from ogb.nodeproppred import DglNodePropPredDataset

def load_oag(global_norm):
    if True:
        hg_orig = dgl.load_graphs('/mnt/OAG_med_dgl.bin')[0][0]
        print(hg_orig)

        subgs = {}
        cnt = 0
        # u, v = hg_orig.all_edges(etype=('paper', 'PF_in_L0', 'field'))
        u, v = hg_orig.all_edges(etype=('paper', 'PF_in_L1', 'field'))
        # u, v = hg_orig.all_edges(etype=('paper', 'PF_in_L2', 'field'))
        field_idx, f_inverse = th.unique(v, return_inverse=True)
        paper_idx, inverse = th.unique(u, return_inverse=True)
        num_class = field_idx.shape[0]
        print(num_class)
        print(paper_idx.shape[0])
        labels = th.zeros((paper_idx.shape[0], num_class)).long()
        labels[inverse, f_inverse] = 1
        print('number of papers in l1 {}'.format(paper_idx.shape[0]))
        print('total links {}|{}'.format(v.shape[0], th.sum(labels)))
        # u, v = hg_orig.all_edges(etype=('paper', 'PF_in_L3', 'field'))
        # u, v = hg_orig.all_edges(etype=('paper', 'PF_in_L4', 'field'))
        # u, v = hg_orig.all_edges(etype=('paper', 'PF_in_L5', 'field'))
        # u, v = hg_orig.all_edges(etype=('paper', 'PF_in_L5', 'field'))

        for etype in hg_orig.canonical_etypes:
            if 'L1' in etype[1] or 'L2' in etype[1] or 'L3' in etype[1] or 'L0' in etype[1] or 'L4' in etype[1] or 'L5' in etype[1]:
                print(etype)
                continue
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)

        th.manual_seed(0)
        rand_idx = th.randperm(paper_idx.shape[0])
        num_train = int(rand_idx.shape[0] * 0.7)
        num_valid = int(rand_idx.shape[0] * 0.1)
        num_test = rand_idx.shape[0] - num_train - num_valid

        train_idx = paper_idx[:num_train]
        val_idx = paper_idx[num_train:num_train+num_valid]
        test_idx = paper_idx[num_train+num_valid:]
        print(labels.shape)
 
        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        print(th.unique(labels))
        category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_class))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))

        # currently we do not support node feature in mag dataset.
        # calculate norm for each edge type and store in edge
        if global_norm is False:
            for canonical_etype in hg.canonical_etypes:
                u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
                _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
                degrees = count[inverse_index]
                norm = th.ones(eid.shape[0]) / degrees
                norm = norm.unsqueeze(1)
                hg.edges[canonical_etype].data['norm'] = norm

        # get target category id
        category_id = len(hg.ntypes)
        for i, ntype in enumerate(hg.ntypes):
            if ntype == category:
                category_id = i

        g = dgl.to_homo(hg)
        if global_norm:
            u, v, eid = g.all_edges(form='all')
            _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = th.ones(eid.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            g.edata['norm'] = norm

        node_ids = th.arange(g.number_of_nodes())
        # find out the target node ids
        node_tids = g.ndata[dgl.NTYPE]
        loc = (node_tids == category_id)
        target_idx = node_ids[loc]
        train_idx = target_idx[train_idx]
        val_idx = target_idx[val_idx]
        test_idx = target_idx[test_idx]
        train_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
        train_mask[train_idx] = True
        val_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
        val_mask[val_idx] = True
        test_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
        test_mask[test_idx] = True
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        g_labels = th.full((g.number_of_nodes(), labels.shape[1]), -1, dtype=th.int)
        g_labels[target_idx[paper_idx]] = labels.int()
        g.ndata['labels'] = g_labels
        return g
    else:
        raise("Do not support other ogbn datasets.")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--global-norm', default=False, action='store_true',
                           help='User global norm instead of per node type norm')
    args = argparser.parse_args()

    start = time.time()
    if os.path.exists('/mnt/oag_l1.bin'):
        g = dgl.load_graphs('/mnt/oag_l1.bin')[0][0]
    else:
        g = load_oag(args.global_norm)

        #dgl.save_graphs('/mnt/oag_l1.bin', [g])
    print('load {} takes {:.3f} seconds'.format('oag', time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(th.sum(g.ndata['train_mask']),
                                                  th.sum(g.ndata['val_mask']),
                                                  th.sum(g.ndata['test_mask'])))

    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    dgl.distributed.partition_graph(g, 'oag', args.num_parts, '/mnt/data',
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges)
