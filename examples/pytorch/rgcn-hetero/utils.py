"""
Utility functions for link prediction
"""

import numpy as np
import torch
import dgl
import os
import pandas as pd

class DrugDataset(object):
    def __init__(self, train_file, valid_file, test_file):
        df_train = pd.read_csv(train_file, sep="\t")
        df_valid = pd.read_csv(valid_file, sep="\t")
        df_test = pd.read_csv(test_file, sep="\t")

        train_triplets = df_train.values.tolist()
        valid_triplets = df_valid.values.tolist()
        test_triplets = df_test.values.tolist()

        def handle_node(node):
            node_info = node.split('::')

            # node_type, node_name
            if len(node_info) == 1:
                return None, None
            return node_info[0], node_info[1]

        def handle_relation(relation):
            return relation

        def get_id(dict, key):
            id = dict.get(key, None)
            if id is None:
                id = len(dict)
                dict[key] = id
            return id

        entity_type_dict = {}
        entity_dict = {}
        relation_dict = {}
        graph_relations = set()

        def handle_triples(triplets):
            heads = []
            head_types = []
            rel_types = []
            tails = []
            tail_types = []
            for triplet in triplets:
                head_type, head_name = handle_node(triplet[0])
                if head_type is None:
                    continue
                tail_type, tail_name = handle_node(triplet[2])
                if tail_type is None:
                    continue
                rel_type = triplet[1]

                rel_id = get_id(relation_dict, rel_type)
                if entity_dict.get(head_type, None) is None:
                    entity_dict[head_type] = {}
                head_id = get_id(entity_dict[head_type], head_name)
                head_type_id = get_id(entity_type_dict, head_type)
                if entity_dict.get(tail_type, None) is None:
                    entity_dict[tail_type] = {}
                tail_id = get_id(entity_dict[tail_type], tail_name)
                tail_type_id = get_id(entity_type_dict, tail_type)
                
                heads.append(head_id)
                head_types.append(head_type_id)
                rel_types.append(rel_id)
                tails.append(tail_id)
                tail_types.append(tail_type_id)
                graph_relations.add((head_type_id, rel_id, tail_type_id))

            heads = np.asarray(heads)
            tails = np.asarray(tails)
            head_types = np.asarray(head_types)
            rel_types = np.asarray(rel_types)
            tail_types = np.asarray(tail_types)
            return heads, tails, head_types, rel_types, tail_types

        train_head, train_tail, train_head_type, train_rel_type, train_tail_type = \
            handle_triples(train_triplets)

        valid_head, valid_tail, valid_head_type, valid_rel_type, valid_tail_type = \
            handle_triples(valid_triplets)

        test_head, test_tail, test_head_type, test_rel_type, test_tail_type = \
            handle_triples(test_triplets)

        self.train = (train_head, train_tail, train_head_type, train_rel_type, train_tail_type)
        self.valid = (valid_head, valid_tail, valid_head_type, valid_rel_type, valid_tail_type)
        self.test = (test_head, test_tail, test_head_type, test_rel_type, test_tail_type)
        self.rel_map = relation_dict
        self.entity_map = entity_dict
        self.entity_type_dict = entity_type_dict
        self.all_rels = list(graph_relations)
        self.num_nodes = {str(node_id): len(entity_dict[key]) for key, node_id in entity_type_dict.items()}

def load_drug_data():
    folder="./data/"
    train_file = os.path.join(folder, "covid19_kg_train.tsv")
    valid_file = os.path.join(folder, "covid19_kg_valid.tsv")
    test_file = os.path.join(folder, "covid19_kg_test.tsv")

    return DrugDataset(train_file, valid_file, test_file)

def build_multi_ntype_graph_from_triplets(num_nodes, all_rels, data_lists, reverse=True):
    """ Create a DGL Hetero graph. 
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    raw_subg = {}

    head_list = []
    tail_list = []
    head_type_list = []
    rel_type_list = []
    tail_type_list = []
    for datas in data_lists:
        head, tail, head_type, rel_type, tail_type = datas
        head_list.append(head)
        tail_list.append(tail)
        head_type_list.append(head_type)
        rel_type_list.append(rel_type)
        tail_type_list.append(tail_type)

    head = np.concatenate(head_list)
    tail = np.concatenate(tail_list)
    head_type = np.concatenate(head_type_list)
    rel_type = np.concatenate(rel_type_list)
    tail_type = np.concatenate(tail_type_list)
        
    for h, t, ht, rt, tt in zip(head, tail, head_type, rel_type, tail_type):
        e_type = (str(ht), str(rt), str(tt))
        if raw_subg.get(e_type, None) is None:
            raw_subg[e_type] = ([], [])

        raw_subg[e_type][0].append(h)
        raw_subg[e_type][1].append(t)
            
        if reverse is True:
            r_type = str(rt + len(all_rels))
            re_type = (str(tt), r_type, str(ht))
            if raw_subg.get(re_type, None) is None:
                raw_subg[re_type] = ([], [])
            raw_subg[re_type][0].append(t)
            raw_subg[re_type][1].append(h)

    subg = []
    for e_type, val in raw_subg.items():
        h_type, r_type, t_type = e_type
        h, t = val
        h = np.asarray(h)
        t = np.asarray(t)

        if h_type == t_type:
            subg.append(dgl.graph((h, t),
                                h_type,
                                r_type,
                                num_nodes=num_nodes[h_type]))
        else:
            subg.append(dgl.bipartite((h, t),
                                      h_type,
                                      r_type,
                                      t_type,
                                      num_nodes=(num_nodes[h_type],
                                                 num_nodes[t_type])))
    g = dgl.hetero_from_relations(subg)

    return g

def build_graph_from_triplets(num_nodes, num_rels, edge_lists, reverse=True):
    """ Create a DGL Hetero graph. 
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    src = []
    rel = []
    dst = []
    raw_subg = {}
    print(num_rels)

    # here there is noly one node type
    s_type = "node"
    d_type = "node"
    edge_list = np.concatenate(edge_lists)
    print(len(edge_list))

    for edge in edge_list:
        s, r, d = edge
        r_type = str(r)
        e_type = (s_type, r_type, d_type)

        if raw_subg.get(e_type, None) is None:
            raw_subg[e_type] = ([], [])
        raw_subg[e_type][0].append(s)
        raw_subg[e_type][1].append(d)

        if reverse is True:
            r_type = str(r + num_rels)
            re_type = (d_type, r_type, s_type)
            if raw_subg.get(re_type, None) is None:
                raw_subg[re_type] = ([], [])
            raw_subg[re_type][0].append(d)
            raw_subg[re_type][1].append(s)

    subg = []
    for e_type, val in raw_subg.items():
        s_type, r_type, d_type = e_type
        s, d = val
        s = np.asarray(s)
        d = np.asarray(d)

        subg.append(dgl.graph((s, d),
                              s_type,
                              r_type,
                              num_nodes=num_nodes))
    g = dgl.hetero_from_relations(subg)

    return g

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function
