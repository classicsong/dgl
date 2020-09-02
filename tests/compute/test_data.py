import os
from pathlib import Path

import dgl.data as data
import unittest, pytest
import numpy as np
import torch as th

def test_minigc():
    ds = data.MiniGCDataset(16, 10, 20)
    g, l = list(zip(*ds))
    print(g, l)

def test_data_hash():
    class HashTestDataset(data.DGLDataset):
        def __init__(self, hash_key=()):
            super(HashTestDataset, self).__init__('hashtest', hash_key=hash_key)
        def _load(self):
            pass

    a = HashTestDataset((True, 0, '1', (1,2,3)))
    b = HashTestDataset((True, 0, '1', (1,2,3)))
    c = HashTestDataset((True, 0, '1', (1,2,4)))
    assert a.hash == b.hash
    assert a.hash != c.hash

def test_row_normalize():
    features = np.array([[1., 1., 1.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([1./3., 1./3., 1./3.]), row_norm_feat)

    features = np.array([[1.], [1.], [1.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1.], [1.], [1.]]), row_norm_feat)

    features = np.array([[1., 0., 0.],[0., 1., 1.],[0., 0., 0.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[0., 0.5, 0.5],[0., 0., 0.]]),
                       row_norm_feat)

    # input (2, 3)
    features = np.array([[1., 0., 0.],[2., 1., 1.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[0.5, 0.25, 0.25]]),
                       row_norm_feat)

    # input (3, 2)
    features = np.array([[1., 0.],[1., 1.],[0., 0.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1., 0.],[0.5, 0.5],[0., 0.]]),
                       row_norm_feat)

def test_col_normalize():
    features = np.array([[1., 1., 1.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[1., 1., 1.]]), col_norm_feat)

    features = np.array([[1.], [1.], [1.]])
    row_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[1./3.],[1./3.], [1./3.]]), row_norm_feat)

    features = np.array([[1., 0., 0.],[1., 1., 0.],[0., 0., 0.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[0.5, 0., 0.],[0.5, 1.0, 0.],[0., 0., 0.]]),
                       col_norm_feat)

    # input (2. 3)
    features = np.array([[1., 0., 0.],[1., 1., 0.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[0.5, 0., 0.],[0.5, 1.0, 0.]]),
                       col_norm_feat)

    # input (3. 2)
    features = np.array([[1., 0.],[1., 1.],[2., 0.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[0.25, 0.],[0.25, 1.0],[0.5, 0.]]),
                       col_norm_feat)

def test_float_row_normalize():
    features = np.array([[1.],[2.],[-3.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1.],[1.],[-1.]]), row_norm_feat)

    features = np.array([[1., 2., -3.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1./6., 2./6., -3./6.]]), row_norm_feat)

    features = np.array([[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[0.5, 0.25, 0.25],[1./6., 2./6., -3./6.]]),
                       row_norm_feat)

     # input (2 3)
    features = np.array([[1., 0., 0.],[-2., 1., 1.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[-0.5, 0.25, 0.25]]),
                       row_norm_feat)

     # input (3, 2)
    features = np.array([[1., 0.],[-2., 1.],[1., 2.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1., 0.],[-2./3., 1./3.],[1./3., 2./3.]]),
                       row_norm_feat)

def test_float_col_normalize():
    features = np.array([[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[1., 1., -1.]]), col_norm_feat)

    features = np.array([[1.], [2.], [-3.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[1./6.],[2./6.], [-3./6.]]), col_norm_feat)

    features = np.array([[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[0.25, 0., 0.],[0.5, 1./3., 0.25],[0.25, 2./3., -0.75]]),
                       col_norm_feat)

    # input (2. 3)
    features = np.array([[1., 0., 0.],[2., 1., -1.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[1./3., 0., 0.],[2./3., 1.0, -1.]]),
                       col_norm_feat)

    # input (3. 2)
    features = np.array([[1., 0.],[2., 1.],[1., -2.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[0.25, 0.],[0.5, 1./3.],[0.25, -2./3.]]),
                       col_norm_feat)

def test_float_col_maxmin_normalize():
    features = np.array([[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 0., 0.]]), col_norm_feat)

    features = np.array([[1.], [2.], [-3.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[4./5.],[5./5.], [0.]]), col_norm_feat)

    features = np.array([[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 0., 3./4.],[1., 0.5, 1.],[0., 1., 0.]]),
                       col_norm_feat)

    # input (2. 3)
    features = np.array([[1., 0., 0.],[2., 1., -1.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 0., 1.],[1., 1., 0.]]),
                       col_norm_feat)

    # input (3. 2)
    features = np.array([[1., 0.],[2., 1.],[4., -2.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 2./3.],[1./3., 1.],[1., 0.]]),
                       col_norm_feat)

@unittest.skip("spacy language test is too heavy")
def test_embed_word2vec():
    import spacy

    inputs = ['hello', 'world']
    languages = ['en_core_web_lg', 'fr_core_news_lg']
    nlps = [spacy.load(languages[0])]

    feats = data.utils.embed_word2vec(inputs[0], nlps)
    doc = nlps[0](inputs[0])
    assert np.allclose(doc.vector, feats)

    nlps.append(spacy.load(languages[1]))
    for input in inputs:
        feats = data.utils.embed_word2vec(input, nlps)
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        assert np.allclose(np.concatenate((doc0.vector, doc1.vector)),
                           feats)

@unittest.skip("spacy language test is too heavy")
def test_parse_lang_feat():
    import spacy

    inputs = ['hello', 'world']
    languages = ['en_core_web_lg', 'fr_core_news_lg']
    nlps = [spacy.load(languages[0]), spacy.load(languages[1])]
    feats = data.utils.parse_lang_feat(inputs, nlps)

    res_feats = []
    for input in inputs:
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        res_feats.append(np.concatenate((doc0.vector, doc1.vector)))
    res_feats = np.stack(res_feats)
    assert np.allclose(feats, res_feats)

    inputs = ["1", "2", "3", "4", "1", "2", "3", "4", "5", "6", "7", "8"]
    feats = data.utils.parse_lang_feat(inputs, nlps)

    res_feats = []
    for input in inputs:
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        res_feats.append(np.concatenate((doc0.vector, doc1.vector)))
    res_feats = np.stack(res_feats)
    assert np.allclose(feats, res_feats)

    inputs = ["1", "2", "3", "4", "1", "2", "3", "4", "5", "6", "7", "8"]
    feats = data.utils.parse_word2vec_feature(inputs, languages)

    res_feats = []
    for input in inputs:
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        res_feats.append(np.concatenate((doc0.vector, doc1.vector)))
    res_feats = np.stack(res_feats)
    assert np.allclose(feats, res_feats)

# @unittest.skip("LabelBinarizer and MultiLabelBinarizer is not included in CI env")
def test_parse_category_feat():
    # single-hot
    inputs = ['A', 'B']
    feats, _ = data.utils.parse_category_single_feat(inputs)
    assert np.allclose(np.array([[1.,0.],[0.,1.]]), feats)

    inputs = ['A', 'B', 'C', 'A']
    feats, _ = data.utils.parse_category_single_feat(inputs)
    assert np.allclose(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]), feats)
    # col norm
    feats, c_map = data.utils.parse_category_single_feat(inputs, norm='col')
    assert np.allclose(np.array([[.5,0.,0.],[0.,1.,0.],[0.,0.,1.],[.5,0.,0.]]), feats)
    assert c_map[0] == 'A'
    assert c_map[1] == 'B'
    assert c_map[2] == 'C'

    # multi-hot
    inputs = [['A'], ['B']]
    feats, _ = data.utils.parse_category_multi_feat(inputs)
    assert np.allclose(np.array([[1.,0.],[0.,1.]]), feats)

    inputs = [['A', 'B', 'C',], ['A', 'B'], ['C'], ['A']]
    feats, c_map = data.utils.parse_category_multi_feat(inputs)
    assert np.allclose(np.array([[1.,1.,1.],[1.,1.,0.],[0.,0.,1.],[1.,0.,0.]]), feats)
    assert c_map[0] == 'A'
    assert c_map[1] == 'B'
    assert c_map[2] == 'C'

    # row norm
    feats, _ = data.utils.parse_category_multi_feat(inputs, norm='row')
    assert np.allclose(np.array([[1./3.,1./3.,1./3.],[.5,.5,0.],[0.,0.,1.],[1.,0.,0.]]), feats)
    # col norm
    feats, _ = data.utils.parse_category_multi_feat(inputs, norm='col')
    assert np.allclose(np.array([[1./3.,0.5,0.5],[1./3.,0.5,0.],[0.,0.,0.5],[1./3.,0.,0.]]), feats)

def test_parse_numerical_feat():
    inputs = [[1., 2., -3.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[1., 1., -1.]]), col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 0., 0.]]), col_norm_feat)

    inputs = [[1.], [2.], [-3.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[1./6.],[2./6.], [-3./6.]]), col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[4./5.],[5./5.], [0.]]), col_norm_feat)

    inputs = [[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[0.25, 0., 0.],[0.5, 1./3., 0.25],[0.25, 2./3., -0.75]]),
                       col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 0., 3./4.],[1., 0.5, 1.],[0., 1., 0.]]),
                       col_norm_feat)

    # input (2. 3)
    inputs = [[1., 0., 0.],[2., 1., -1.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[1./3., 0., 0.],[2./3., 1.0, -1.]]),
                       col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 0., 1.],[1., 1., 0.]]),
                       col_norm_feat)

    # input (3. 2)
    inputs = [[1., 0.],[2., 1.],[1., -2.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[0.25, 0.],[0.5, 1./3.],[0.25, -2./3.]]),
                       col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 2./3.],[1., 1.],[0., 0.]]),
                       col_norm_feat)

def test_parse_numerical_multihot_feat():
    inputs = [0., 15., 20., 10.1, 25., 40.]
    low = 10.
    high = 30.
    bucket_cnt = 2 #10~20, 20~30
    window_size = 0.
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size)
    assert np.allclose(np.array([[1., 0.], [1., 0.], [0., 1.], [1., 0.], [0., 1.], [0., 1.]]), feat)

    inputs = [0., 5., 15., 20., 10.1, 25., 30.1, 40.]
    low = 10.
    high = 30.
    bucket_cnt = 4 #10~15,15~20,20~25,25~30
    window_size = 10.
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size)
    assert np.allclose(np.array([[1., 0., 0., 0],
                                 [1., 0., 0., 0],
                                 [1., 1., 1., 0.],
                                 [0., 1., 1., 1.],
                                 [1., 1., 0., 0.],
                                 [0., 0., 1., 1.],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., 1.]]), feat)

    # col norm
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size, norm='col')
    assert np.allclose(np.array([[1./4., 0.,    0.,    0],
                                 [1./4., 0.,    0.,    0],
                                 [1./4., 1./3., 1./3., 0.],
                                 [0.,    1./3., 1./3., 1./4.],
                                 [1./4., 1./3., 0.,    0.],
                                 [0.,    0.,    1./3., 1./4.],
                                 [0.,    0.,    0.,    1./4.],
                                 [0.,    0.,    0.,    1./4.]]), feat)

    # row norm
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size, norm='row')
    assert np.allclose(np.array([[1., 0., 0., 0],
                                 [1., 0., 0., 0],
                                 [1./3., 1./3., 1./3., 0.],
                                 [0., 1./3., 1./3., 1./3.],
                                 [1./2., 1./2., 0., 0.],
                                 [0., 0., 1./2., 1./2.],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., 1.]]), feat)

def create_category_node_feat(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3\n".format(separator,separator,separator))
    node_feat_f.write("node1{}A{}B{}A,B\n".format(separator,separator,separator))
    node_feat_f.write("node2{}A{}{}A\n".format(separator,separator,separator))
    node_feat_f.write("node3{}C{}B{}C,B\n".format(separator,separator,separator))
    node_feat_f.write("node3{}A{}C{}A,C\n".format(separator,separator,separator))
    node_feat_f.close()

def create_numerical_node_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3{}feat4\n".format(sep,sep,sep,sep))
    node_feat_f.write("node1{}1.{}2.{}0.{}1.,2.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.write("node2{}2.{}-1.{}0.{}2.,-1.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.write("node3{}0.{}0.{}0.{}0.,0.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.write("node3{}4.{}-2.{}0.{}4.,-2.,0.\n".format(sep,sep,sep,sep))
    node_feat_f.close()

def create_numerical_bucket_node_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}0.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}5.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}15.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}20.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}10.1\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}25.\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}30.1\n".format(sep,sep,sep))
    node_feat_f.write("node1{}0{}40.\n".format(sep,sep,sep))
    node_feat_f.close()

def create_numerical_edge_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_s{}node_d{}feat1\n".format(sep,sep))
    node_feat_f.write("node1{}node4{}1.\n".format(sep,sep))
    node_feat_f.write("node2{}node5{}2.\n".format(sep,sep))
    node_feat_f.write("node3{}node6{}0.\n".format(sep,sep))
    node_feat_f.write("node3{}node3{}4.\n".format(sep,sep))
    node_feat_f.close()

def create_word_node_feat(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3\n".format(separator,separator,separator))
    node_feat_f.write("node1{}A{}B{}24\n".format(separator,separator,separator))
    node_feat_f.write("node2{}A{}{}1\n".format(separator,separator,separator))
    node_feat_f.write("node3{}C{}B{}12\n".format(separator,separator,separator))
    node_feat_f.write("node3{}A{}C{}13\n".format(separator,separator,separator))
    node_feat_f.close()

def create_multiple_node_feat(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}feat1{}feat2{}feat3\n".format(separator,separator,separator))
    node_feat_f.write("node1{}A{}0.1{}A,B\n".format(separator,separator,separator))
    node_feat_f.write("node2{}A{}0.3{}A\n".format(separator,separator,separator))
    node_feat_f.write("node3{}C{}0.2{}C,B\n".format(separator,separator,separator))
    node_feat_f.write("node4{}A{}-1.1{}A,C\n".format(separator,separator,separator))
    node_feat_f.close()

def create_multiple_edge_feat(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_s{}node_d{}feat1{}feat2{}feat3\n".format(sep,sep,sep,sep))
    node_feat_f.write("node1{}node_a{}0.2{}0.1{}1.1\n".format(sep,sep,sep,sep))
    node_feat_f.write("node2{}node_b{}-0.3{}0.3{}1.2\n".format(sep,sep,sep,sep))
    node_feat_f.write("node3{}node_c{}0.3{}0.2{}-1.2\n".format(sep,sep,sep,sep))
    node_feat_f.write("node4{}node_d{}-0.2{}-1.1{}0.9\n".format(sep,sep,sep,sep))
    node_feat_f.close()

def create_node_labels(tmpdir, file_name, separator='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node{}label1{}label2\n".format(separator,separator))
    node_feat_f.write("node1{}A{}D,A\n".format(separator,separator))
    node_feat_f.write("node2{}A{}E,C,D\n".format(separator,separator))
    node_feat_f.write("node3{}C{}F,A,B\n".format(separator,separator))
    node_feat_f.write("node4{}A{}G,E\n".format(separator,separator))
    node_feat_f.close()

def create_edge_labels(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_0{}node_1{}label1{}label2\n".format(sep,sep,sep))
    node_feat_f.write("node1{}node4{}A{}D,A\n".format(sep,sep,sep))
    node_feat_f.write("node2{}node3{}A{}E,C,D\n".format(sep,sep,sep))
    node_feat_f.write("node3{}node2{}C{}F,A,B\n".format(sep,sep,sep))
    node_feat_f.write("node4{}node1{}A{}G,E\n".format(sep,sep,sep))
    node_feat_f.close()

def create_graph_edges(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write("node_0{}node_1{}rel_1{}rel_2\n".format(sep,sep,sep))
    node_feat_f.write("node1{}node2{}A{}C\n".format(sep,sep,sep))
    node_feat_f.write("node2{}node1{}A{}C\n".format(sep,sep,sep))
    node_feat_f.write("node3{}node1{}A{}C\n".format(sep,sep,sep))
    node_feat_f.write("node4{}node3{}A{}B\n".format(sep,sep,sep))
    node_feat_f.write("node4{}node4{}A{}A\n".format(sep,sep,sep))
    node_feat_f.close()

def create_multiple_label(tmpdir, file_name, sep='\t'):
    node_feat_f = open(os.path.join(tmpdir, file_name), "w")
    node_feat_f.write(
        "node{}label1{}label2{}label3{}label4{}label5{}node_d{}node_d2{}node_d3\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node1{}A{}A{}C{}A,B{}A,C{}node3{}node1{}node4\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node2{}B{}B{}B{}A{}B{}node4{}node2{}node5\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node3{}C{}C{}A{}C,B{}A{}node5{}node1{}node6\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.write("node4{}A{}A{}A{}A,C{}A,B{}node6{}node2{}node7\n".format(
        sep,sep,sep,sep,sep,sep,sep,sep))
    node_feat_f.close()

# @unittest.skip("LabelBinarizer and MultiLabelBinarizer is not included in CI env")
def test_node_category_feature_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_category_node_feat(Path(tmpdirname), 'node_category_feat.csv')

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_category_feat.csv'))
        feat_loader.addCategoryFeature([0, 1], feat_name='tf')
        feat_loader.addCategoryFeature(['node', 'feat1'], norm='row', node_type='node')
        feat_loader.addCategoryFeature(['node', 'feat1'], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,0],[1,0],[0,1],[1,0]]),
                           f_1[3])
        assert np.allclose(np.array([[1,0],[1,0],[0,1],[1,0]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,0],[1./3.,0],[0,1],[1./3.,0]]),
                           f_3[3])

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_category_feat.csv'))
        feat_loader.addCategoryFeature([0, 1, 2])
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'], norm='row', node_type='node')
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'nf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0,0.5,0.5],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1./2.,0],
                                     [1./3.,0,    0],
                                     [0,    1./2.,1./2.],
                                     [1./3.,0,    1./2.]]),
                           f_3[3])

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_category_feat.csv'))
        feat_loader.addCategoryFeature([0, 1, 2], rows=[0,1,3])
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'],
                                        rows=[0,1,3], norm='row', node_type='node')
        feat_loader.addCategoryFeature(['node', 'feat1', 'feat2'],
                                        rows=[0,1,3], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1.,0.],
                                     [1./3.,0.,0.],
                                     [1./3.,0.,1.]]),
                           f_3[3])


        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                                'node_category_feat.csv'))
        feat_loader.addMultiCategoryFeature([0, 3], separator=',')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',', norm='row', node_type='node')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',', norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0,0.5,0.5],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1./2.,0],
                                     [1./3.,0,    0],
                                     [0,    1./2.,1./2.],
                                     [1./3.,0,    1./2.]]),
                           f_3[3])

        feat_loader.addMultiCategoryFeature([0, 3], rows=[0,1,3], separator=',')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',',
                                            rows=[0,1,3], norm='row', node_type='node')
        feat_loader.addMultiCategoryFeature(['node', 'feat3'], separator=',',
                                            rows=[0,1,3], norm='col', node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1,1,0],[1,0,0],[1,0,1]]),
                           f_1[3])
        assert np.allclose(np.array([[0.5,0.5,0],[1,0,0],[0.5,0,0.5]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3.,1.,0.],
                                     [1./3.,0.,0.],
                                     [1./3.,0.,1.]]),
                           f_3[3])

def test_node_numerical_feature_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_numerical_node_feat(Path(tmpdirname), 'node_numerical_feat.csv')

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_feat.csv'))
        feat_loader.addNumericalFeature([0, 1])
        feat_loader.addNumericalFeature(['node', 'feat1'], norm='standard', node_type='node')
        feat_loader.addNumericalFeature(['node', 'feat1'], norm='min-max', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'nf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1.],[2.],[0.],[4.]]),
                           f_1[3])
        assert np.allclose(np.array([[1./7.],[2./7.],[0.],[4./7.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4.],[2./4],[0.],[1.]]),
                           f_3[3])

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_feat.csv'))
        feat_loader.addNumericalFeature([0,1,2,3],feat_name='tf')
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        norm='standard',
                                        node_type='node')
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        norm='min-max',
                                        node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1.,2.,0.],[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[1./7.,2./5.,0.],[2./7.,-1./5.,0.],[0.,0.,0.],[4./7.,-2./5.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4.,1.,0.],[2./4,1./4.,0.],[0.,2./4.,0.],[1.,0.,0.]]),
                           f_3[3])

        feat_loader.addNumericalFeature([0,1,2,3],rows=[1,2,3])
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        rows=[1,2,3],
                                        norm='standard',
                                        node_type='node')
        feat_loader.addNumericalFeature(['node', 'feat1','feat2','feat3'],
                                        rows=[1,2,3],
                                        norm='min-max',
                                        node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[2./6.,-1./3.,0.],[0.,0.,0.],[4./6.,-2./3.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[2./4.,1./2.,0.],[0.,1.,0.],[1.,0.,0.]]),
                           f_3[3])

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_feat.csv'))
        feat_loader.addMultiNumericalFeature([0,4], separator=',')
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             norm='standard',
                                             node_type='node')
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             norm='min-max',
                                             node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1.,2.,0.],[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[1./7.,2./5.,0.],[2./7.,-1./5.,0.],[0.,0.,0.],[4./7.,-2./5.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4.,1.,0.],[2./4,1./4.,0.],[0.,2./4.,0.],[1.,0.,0.]]),
                           f_3[3])

        feat_loader.addMultiNumericalFeature([0,4], separator=',', rows=[1,2,3])
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             rows=[1,2,3],
                                             norm='standard',
                                             node_type='node')
        feat_loader.addMultiNumericalFeature(['node', 'feat4'],
                                             separator=',',
                                             rows=[1,2,3],
                                             norm='min-max',
                                             node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[2.,-1.,0.],[0.,0.,0.],[4.,-2.,0.]]),
                           f_1[3])
        assert np.allclose(np.array([[2./6.,-1./3.,0.],[0.,0.,0.],[4./6.,-2./3.,0.]]),
                           f_2[3])
        assert np.allclose(np.array([[2./4.,1./2.,0.],[0.,1.,0.],[1.,0.,0.]]),
                           f_3[3])

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_numerical_bucket_node_feat(Path(tmpdirname), 'node_numerical_bucket_feat.csv')

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_bucket_feat.csv'))
        feat_loader.addNumericalBucketFeature([0, 2],
                                              feat_name='tf',
                                              range=[10,30],
                                              bucket_cnt=2)
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='row', node_type='node')
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.],
                                    [1., 0.], [0., 1.], [0., 1.], [0., 1.]]),
                           f_1[3])
        assert np.allclose(np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.],
                                    [1., 0.], [0., 1.], [0., 1.], [0., 1.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4., 0.], [1./4., 0.], [1./4., 0.], [0., 1./4],
                                     [1./4., 0.], [0., 1./4.], [0., 1./4.], [0., 1./4.]]),
                           f_3[3])

        feat_loader.addNumericalBucketFeature([0, 2],
                                              rows=[0,2,3,4,5,6],
                                              range=[10,30],
                                              bucket_cnt=2)
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              rows=[0,2,3,4,5,6],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='row', node_type='node')
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              rows=[0,2,3,4,5,6],
                                              range=[10,30],
                                              bucket_cnt=2,
                                              norm='col', node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1., 0.], [1., 0.], [0., 1.],
                                     [1., 0.], [0., 1.], [0., 1.]]),
                           f_1[3])
        assert np.allclose(np.array([[1., 0.], [1., 0.], [0., 1.],
                                     [1., 0.], [0., 1.], [0., 1.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./3., 0.], [1./3., 0.], [0., 1./3],
                                     [1./3., 0.], [0., 1./3.], [0., 1./3.]]),
                           f_3[3])

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_numerical_bucket_feat.csv'))
        feat_loader.addNumericalBucketFeature([0, 2],
                                              feat_name='tf',
                                              range=[10,30],
                                              bucket_cnt=4,
                                              slide_window_size=10.)
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=4,
                                              slide_window_size=10.,
                                              norm='row', node_type='node')
        feat_loader.addNumericalBucketFeature(['node', 'feat2'],
                                              range=[10,30],
                                              bucket_cnt=4,
                                              slide_window_size=10.,
                                              norm='col', node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(np.array([[1., 0., 0., 0],
                                     [1., 0., 0., 0],
                                     [1., 1., 1., 0.],
                                     [0., 1., 1., 1.],
                                     [1., 1., 0., 0.],
                                     [0., 0., 1., 1.],
                                     [0., 0., 0., 1.],
                                     [0., 0., 0., 1.]]),
                           f_1[3])
        assert np.allclose(np.array([[1., 0., 0., 0],
                                     [1., 0., 0., 0],
                                     [1./3., 1./3., 1./3., 0.],
                                     [0., 1./3., 1./3., 1./3.],
                                     [1./2., 1./2., 0., 0.],
                                     [0., 0., 1./2., 1./2.],
                                     [0., 0., 0., 1.],
                                     [0., 0., 0., 1.]]),
                           f_2[3])
        assert np.allclose(np.array([[1./4., 0.,    0.,    0],
                                     [1./4., 0.,    0.,    0],
                                     [1./4., 1./3., 1./3., 0.],
                                     [0.,    1./3., 1./3., 1./4.],
                                     [1./4., 1./3., 0.,    0.],
                                     [0.,    0.,    1./3., 1./4.],
                                     [0.,    0.,    0.,    1./4.],
                                     [0.,    0.,    0.,    1./4.]]),
                           f_3[3])

def test_edge_numerical_feature_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_numerical_edge_feat(Path(tmpdirname), 'edge_numerical_feat.csv')

        feat_loader = data.EdgeFeatureLoader(os.path.join(tmpdirname,
                                                          'edge_numerical_feat.csv'))
        feat_loader.addNumericalFeature([0, 1, 2], feat_name='tf')
        feat_loader.addNumericalFeature(['node_s', 'node_d', 'feat1'],
                                        norm='standard',
                                        edge_type=('src', 'rel', 'dst'))
        feat_loader.addNumericalFeature(['node_d', 'node_s', 'feat1'],
                                        norm='min-max',
                                        edge_type=('dst', 'rev-rel', 'src'))
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'ef'
        assert f_3[0] == 'ef'
        assert f_1[1] is None
        assert f_2[1] == ('src', 'rel', 'dst')
        assert f_3[1] == ('dst', 'rev-rel', 'src')
        assert f_1[2] == f_2[2]
        assert f_1[2] == ['node1','node2','node3','node3']
        assert f_3[2] == ['node4','node5','node6','node3']
        assert f_1[3] == f_2[3]
        assert f_1[3] == ['node4','node5','node6','node3']
        assert f_3[3] == ['node1','node2','node3','node3']
        assert np.allclose(np.array([[1.],[2.],[0.],[4.]]),
                           f_1[4])
        assert np.allclose(np.array([[1./7.],[2./7.],[0.],[4./7.]]),
                           f_2[4])
        assert np.allclose(np.array([[1./4.],[2./4],[0.],[1.]]),
                           f_3[4])
        feat_loader.addNumericalFeature(['node_s', 'node_d', 'feat1'],
                                        rows=[1,2,3],
                                        norm='standard',
                                        edge_type=('src', 'rel', 'dst'))
        feat_loader.addNumericalFeature(['node_d', 'node_s', 'feat1'],
                                        rows=[1,2,3],
                                        norm='min-max',
                                        edge_type=('dst', 'rev-rel', 'src'))
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        assert f_1[1] == ('src', 'rel', 'dst')
        assert f_2[1] == ('dst', 'rev-rel', 'src')
        assert f_1[2] == ['node2','node3','node3']
        assert f_2[2] == ['node5','node6','node3']
        assert f_1[3] == ['node5','node6','node3']
        assert f_2[3] == ['node2','node3','node3']
        assert np.allclose(np.array([[2./6.],[0.],[4./6.]]),
                           f_1[4])
        assert np.allclose(np.array([[2./4],[0.],[1.]]),
                           f_2[4])

# @unittest.skip("spacy language test is too heavy")
def test_node_word2vec_feature_loader():
    import tempfile
    import spacy
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_word_node_feat(Path(tmpdirname), 'node_word_feat.csv')

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_word_feat.csv'))
        feat_loader.addWord2VecFeature([0, 1], languages=['en_core_web_lg'], feat_name='tf')
        feat_loader.addWord2VecFeature(['node', 'feat1'],
                                       languages=['en_core_web_lg'],
                                       node_type='node')
        feat_loader.addWord2VecFeature(['node', 'feat1'],
                                       languages=['en_core_web_lg'],
                                       node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[0] == 'tf'
        assert f_2[0] == 'nf'
        assert f_3[0] == 'nf'
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(f_1[3], f_2[3])
        assert np.allclose(f_1[3], f_3[3])
        nlp = spacy.load('en_core_web_lg')
        assert np.allclose(np.array([nlp("A").vector,
                                     nlp("A").vector,
                                     nlp("C").vector,
                                     nlp("A").vector]),
                           f_1[3])

        feat_loader.addWord2VecFeature([0, 3], languages=['en_core_web_lg', 'fr_core_news_lg'])
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        f_1 = feat_loader._raw_features[3]
        f_2 = feat_loader._raw_features[4]
        f_3 = feat_loader._raw_features[5]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(f_1[3], f_2[3])
        assert np.allclose(f_1[3], f_3[3])
        nlp1 = spacy.load('fr_core_news_lg')
        assert np.allclose(np.array([np.concatenate((nlp("24").vector, nlp1("24").vector)),
                                     np.concatenate((nlp("1").vector, nlp1("1").vector)),
                                     np.concatenate((nlp("12").vector, nlp1("12").vector)),
                                     np.concatenate((nlp("13").vector, nlp1("13").vector))]),
                           f_1[3])

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_word_feat.csv'))
        feat_loader.addWord2VecFeature([0, 3],
                                       rows=[1,2],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'])
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       rows=[1,2],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        feat_loader.addWord2VecFeature(['node', 'feat3'],
                                       rows=[1,2],
                                       languages=['en_core_web_lg', 'fr_core_news_lg'],
                                       node_type='node')
        f_1 = feat_loader._raw_features[0]
        f_2 = feat_loader._raw_features[1]
        f_3 = feat_loader._raw_features[2]
        assert f_1[1] is None
        assert f_2[1] == 'node'
        assert f_3[1] == 'node'
        assert f_1[2] == f_2[2]
        assert f_1[2] == f_3[2]
        assert np.allclose(f_1[3], f_2[3])
        assert np.allclose(f_1[3], f_3[3])
        nlp1 = spacy.load('fr_core_news_lg')
        assert np.allclose(np.array([np.concatenate((nlp("1").vector, nlp1("1").vector)),
                                     np.concatenate((nlp("12").vector, nlp1("12").vector))]),
                           f_1[3])

def test_node_label_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_node_labels(Path(tmpdirname), 'labels.csv')
        label_loader = data.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'labels.csv'))
        label_loader.addTrainSet([0,1])
        label_loader.addValidSet(['node','label1'], node_type='node')
        label_loader.addTestSet(['node','label1'], rows=[0,2], node_type='node')
        label_loader.addSet(['node','label1'], [0.5, 0.25, 0.25], rows=[0,1,2,3], node_type='nt')
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == 'node'
        assert l_3[0] == 'node'
        assert l_4[0] == 'nt'
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert np.allclose(l_1[2], l_2[2])
        assert np.allclose(l_1[2], np.array([[1.,0.,], [1.,0.], [0.,1.],[1.,0.]]))
        assert np.allclose(l_3[2], np.array([[1.,0.], [0.,1.]]))
        assert np.allclose(l_4[2], l_1[2])
        assert l_1[3] == (1., 0., 0.)
        assert l_2[3] == (0., 1., 0.)
        assert l_3[3] == (0., 0., 1.)
        assert l_4[3] == (0.5, 0.25, 0.25)

        label_loader = data.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'labels.csv'))
        label_loader.addTrainSet([0,2], multilabel=True, separator=',')
        label_loader.addValidSet(['node','label2'],
                                 multilabel=True,
                                 separator=',',
                                 node_type='node')
        label_loader.addTestSet(['node','label2'],
                                 multilabel=True,
                                 separator=',',
                                 rows=[0,2],
                                 node_type='node')
        label_loader.addSet(['node','label2'],
                            [0.5, 0.25, 0.25],
                            multilabel=True,
                            separator=',', rows=[0,1,2,3], node_type='nt')
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == 'node'
        assert l_3[0] == 'node'
        assert l_4[0] == 'nt'
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert np.allclose(l_1[2], l_2[2])
        assert np.allclose(l_1[2], np.array([[1.,0.,0.,1.,0.,0.,0.],
                                             [0.,0.,1.,1.,1.,0.,0.],
                                             [1.,1.,0.,0.,0.,1.,0.],
                                             [0.,0.,0.,0.,1.,0.,1.]]))
        assert np.allclose(l_3[2], np.array([[1.,0.,1.,0.],
                                             [1.,1.,0.,1.]]))
        assert np.allclose(l_4[2], l_1[2])
        assert l_1[3] == (1., 0., 0.)
        assert l_2[3] == (0., 1., 0.)
        assert l_3[3] == (0., 0., 1.)
        assert l_4[3] == (0.5, 0.25, 0.25)


def test_edge_label_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_edge_labels(Path(tmpdirname), 'edge_labels.csv')
        label_loader = data.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_labels.csv'))
        label_loader.addTrainSet([0,1,2])
        label_loader.addValidSet(['node_0','node_1','label1'],
                                 edge_type=('src','rel','dst'))
        label_loader.addTestSet(['node_0','node_1','label1'],
                                rows=[0,2],
                                edge_type=('src','rel','dst'))
        label_loader.addSet(['node_0','node_1','label1'],
                            [0.5, 0.25, 0.25],
                            rows=[0,1,2,3],
                            edge_type=('src_n','rel_r','dst_n'))
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == ('src','rel','dst')
        assert l_3[0] == ('src','rel','dst')
        assert l_4[0] == ('src_n','rel_r','dst_n')
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert l_1[2] == l_2[2]
        assert l_1[2] == ['node4', 'node3', 'node2', 'node1']
        assert l_3[2] == ['node4', 'node2']
        assert l_4[2] == l_1[2]
        assert np.allclose(l_1[3], l_2[3])
        assert np.allclose(l_1[3], np.array([[1.,0.,], [1.,0.], [0.,1.],[1.,0.]]))
        assert np.allclose(l_3[3], np.array([[1.,0.], [0.,1.]]))
        assert np.allclose(l_4[3], l_1[3])
        assert l_1[4] == (1., 0., 0.)
        assert l_2[4] == (0., 1., 0.)
        assert l_3[4] == (0., 0., 1.)
        assert l_4[4] == (0.5, 0.25, 0.25)

        label_loader = data.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_labels.csv'))
        label_loader.addTrainSet([0,1,3], multilabel=True, separator=',')
        label_loader.addValidSet(['node_0','node_1','label2'],
                                 multilabel=True,
                                 separator=',',
                                 edge_type=('src','rel','dst'))
        label_loader.addTestSet(['node_0','node_1','label2'],
                                 multilabel=True,
                                 separator=',',
                                 rows=[0,2],
                                 edge_type=('src','rel','dst'))
        label_loader.addSet(['node_0','node_1','label2'],
                            [0.5, 0.25, 0.25],
                            multilabel=True,
                            separator=',',
                            rows=[0,1,2,3],
                            edge_type=('src_n','rel_r','dst_n'))
        l_1 = label_loader._labels[0]
        l_2 = label_loader._labels[1]
        l_3 = label_loader._labels[2]
        l_4 = label_loader._labels[3]
        assert l_1[0] == None
        assert l_2[0] == ('src','rel','dst')
        assert l_3[0] == ('src','rel','dst')
        assert l_4[0] == ('src_n','rel_r','dst_n')
        assert l_1[1] == l_2[1]
        assert l_1[1] == ['node1', 'node2', 'node3', 'node4']
        assert l_3[1] == ['node1', 'node3']
        assert l_4[1] == l_1[1]
        assert l_1[2] == l_2[2]
        assert l_1[2] == ['node4', 'node3', 'node2', 'node1']
        assert l_3[2] == ['node4', 'node2']
        assert l_4[2] == l_1[2]
        assert np.allclose(l_1[3], l_2[3])
        assert np.allclose(l_1[3], np.array([[1.,0.,0.,1.,0.,0.,0.],
                                             [0.,0.,1.,1.,1.,0.,0.],
                                             [1.,1.,0.,0.,0.,1.,0.],
                                             [0.,0.,0.,0.,1.,0.,1.]]))
        assert np.allclose(l_3[3], np.array([[1.,0.,1.,0.],
                                             [1.,1.,0.,1.]]))
        assert np.allclose(l_4[3], l_1[3])
        assert l_1[4] == (1., 0., 0.)
        assert l_2[4] == (0., 1., 0.)
        assert l_3[4] == (0., 0., 1.)
        assert l_4[4] == (0.5, 0.25, 0.25)

def test_edge_loader():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'graphs.csv')
        edge_loader = data.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addEdges([0,1])
        edge_loader.addEdges(['node_0','node_1'])
        edge_loader.addEdges(['node_0','node_1'],
                             rows=np.array([1,2,3,4]),
                             edge_type=('src', 'edge', 'dst'))
        e_1 = edge_loader._edges[0]
        e_2 = edge_loader._edges[1]
        e_3 = edge_loader._edges[2]
        assert e_1[0] == None
        assert e_2[0] == None
        assert e_3[0] == ('src','edge','dst')
        assert e_1[1] == e_2[1]
        assert e_1[1] == ['node1', 'node2', 'node3', 'node4', 'node4']
        assert e_3[1] == ['node2', 'node3', 'node4', 'node4']
        assert e_1[2] == e_2[2]
        assert e_1[2] == ['node2', 'node1', 'node1', 'node3', 'node4']
        assert e_3[2] == ['node1', 'node1', 'node3', 'node4']

        edge_loader = data.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addCategoryRelationEdge([0,1,2],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_1'],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_1'],
                                            rows=np.array([1,2,3,4]),
                                            src_type='src',
                                            dst_type='dst')
        e_1 = edge_loader._edges[0]
        e_2 = edge_loader._edges[1]
        e_3 = edge_loader._edges[2]
        assert e_1[0] == ('src_t','A','dst_t')
        assert e_2[0] == ('src_t','A','dst_t')
        assert e_3[0] == ('src','A','dst')
        assert e_1[1] == e_2[1]
        assert e_1[1] == ['node1', 'node2', 'node3', 'node4', 'node4']
        assert e_3[1] == ['node2', 'node3', 'node4', 'node4']
        assert e_1[2] == e_2[2]
        assert e_1[2] == ['node2', 'node1', 'node1', 'node3', 'node4']
        assert e_3[2] == ['node1', 'node1', 'node3', 'node4']

        edge_loader = data.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addCategoryRelationEdge([0,1,3],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_2'],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_2'],
                                            rows=np.array([1,2,3,4]),
                                            src_type='src',
                                            dst_type='dst')
        e_1 = edge_loader._edges[0]
        e_2 = edge_loader._edges[1]
        e_3 = edge_loader._edges[2]
        assert e_1[0] == ('src_t','C','dst_t')
        assert e_2[0] == ('src_t','B','dst_t')
        assert e_3[0] == ('src_t','A','dst_t')
        e_4 = edge_loader._edges[3]
        e_5 = edge_loader._edges[4]
        e_6 = edge_loader._edges[5]
        assert e_4[0] == ('src_t','C','dst_t')
        assert e_5[0] == ('src_t','B','dst_t')
        assert e_6[0] == ('src_t','A','dst_t')
        assert e_1[1] == e_4[1]
        assert e_2[1] == e_5[1]
        assert e_3[1] == e_6[1]
        assert e_1[1] == ['node1', 'node2', 'node3']
        assert e_2[1] == ['node4']
        assert e_3[1] == ['node4']
        assert e_1[2] == e_4[2]
        assert e_2[2] == e_5[2]
        assert e_3[2] == e_6[2]
        assert e_1[2] == ['node2', 'node1', 'node1']
        assert e_2[2] == ['node3']
        assert e_3[2] == ['node4']
        e_7 = edge_loader._edges[6]
        e_8 = edge_loader._edges[7]
        e_9 = edge_loader._edges[8]
        assert e_7[0] == ('src','C','dst')
        assert e_8[0] == ('src','B','dst')
        assert e_9[0] == ('src','A','dst')
        assert e_7[1] == ['node2', 'node3']
        assert e_8[1] == ['node4']
        assert e_9[1] == ['node4']
        assert e_7[2] == ['node1', 'node1']
        assert e_8[2] == ['node3']
        assert e_9[2] == ['node4']

def test_node_feature_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_node_feat(Path(tmpdirname), 'node_feat.csv')

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_feat.csv'))
        feat_loader.addNumericalFeature([0,2],norm='standard')
        feat_loader.addCategoryFeature([0,1])
        feat_loader.addMultiCategoryFeature([0,3], separator=',')

        node_dicts = {}
        result = feat_loader.process(node_dicts)
        assert len(result) == 1
        nids, feats = result[None]['nf']
        assert np.allclose(np.array([0,1,2,3]), nids)
        assert np.allclose(np.concatenate([np.array([[0.1/1.7],[0.3/1.7],[0.2/1.7],[-1.1/1.7]]),
                                           np.array([[1.,0.],[1.,0.],[0.,1.],[1.,0.]]),
                                           np.array([[1.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.]])],
                                           axis=1),
                           feats)
        assert node_dicts[None]['node1'] == 0
        assert node_dicts[None]['node2'] == 1
        assert node_dicts[None]['node3'] == 2
        assert node_dicts[None]['node4'] == 3
        node_dicts = {None: {'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        nids, feats = result[None]['nf']
        assert np.allclose(np.array([3,2,1,0]), nids)
        assert np.allclose(np.concatenate([np.array([[0.1/1.7],[0.3/1.7],[0.2/1.7],[-1.1/1.7]]),
                                           np.array([[1.,0.],[1.,0.],[0.,1.],[1.,0.]]),
                                           np.array([[1.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.]])],
                                           axis=1),
                           feats)

        feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname,
                                                          'node_feat.csv'))
        feat_loader.addCategoryFeature(['node','feat1'], node_type='n1')
        feat_loader.addMultiCategoryFeature(['node','feat3'], separator=',', node_type='n1')
        feat_loader.addNumericalFeature(['node','feat2'], norm='standard', node_type='n2')
        node_dicts = {'n2':{'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        assert len(result) == 2
        assert len(node_dicts) == 2
        nids, feats = result['n1']['nf']
        assert np.allclose(np.array([0,1,2,3]), nids)
        assert np.allclose(np.concatenate([np.array([[1.,0.],[1.,0.],[0.,1.],[1.,0.]]),
                                           np.array([[1.,1.,0.],[1.,0.,0.],[0.,1.,1.],[1.,0.,1.]])],
                                           axis=1),
                           feats)
        nids, feats = result['n2']['nf']
        assert np.allclose(np.array([3,2,1,0]), nids)
        assert np.allclose(np.array([[0.1/1.7],[0.3/1.7],[0.2/1.7],[-1.1/1.7]]),
                           feats)

def test_edge_feature_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_edge_feat(Path(tmpdirname), 'edge_feat.csv')

        feat_loader = data.EdgeFeatureLoader(os.path.join(tmpdirname,
                                                          'edge_feat.csv'))
        feat_loader.addNumericalFeature([0,1,2],norm='standard')
        feat_loader.addNumericalFeature([0,1,3],norm='min-max')
        feat_loader.addNumericalFeature([0,1,4])
        node_dicts = {}
        result = feat_loader.process(node_dicts)
        assert len(result) == 1
        snids, dnids, feats = result[None]['ef']
        assert np.allclose(np.array([0,1,2,3]), snids)
        assert np.allclose(np.array([4,5,6,7]), dnids)
        assert np.allclose(np.concatenate([np.array([[0.2/1.0],[-0.3/1.0],[0.3/1.0],[-0.2/1.0]]),
                                           np.array([[1.2/1.4],[1.0],[1.3/1.4],[0.]]),
                                           np.array([[1.1],[1.2],[-1.2],[0.9]])],
                                           axis=1),
                           feats)
        assert node_dicts[None]['node1'] == 0
        assert node_dicts[None]['node2'] == 1
        assert node_dicts[None]['node3'] == 2
        assert node_dicts[None]['node4'] == 3
        node_dicts = {None: {'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        snids, dnids, feats = result[None]['ef']
        assert np.allclose(np.array([3,2,1,0]), snids)
        assert np.allclose(np.array([4,5,6,7]), dnids)
        assert np.allclose(np.concatenate([np.array([[0.2/1.0],[-0.3/1.0],[0.3/1.0],[-0.2/1.0]]),
                                           np.array([[1.2/1.4],[1.0],[1.3/1.4],[0.]]),
                                           np.array([[1.1],[1.2],[-1.2],[0.9]])],
                                           axis=1),
                           feats)

        feat_loader = data.EdgeFeatureLoader(os.path.join(tmpdirname,
                                                          'edge_feat.csv'))
        feat_loader.addNumericalFeature([0,1,2],norm='standard',edge_type=('n0','r0','n1'))
        feat_loader.addNumericalFeature([0,1,3],norm='min-max',edge_type=('n0','r0','n1'))
        feat_loader.addNumericalFeature([0,1,4],edge_type=('n1','r1','n0'))
        node_dicts = {'n0':{'node1':3,
                             'node2':2,
                             'node3':1,
                             'node4':0}}
        result = feat_loader.process(node_dicts)
        assert len(result) == 2
        snids, dnids, feats = result[('n0','r0','n1')]['ef']
        assert np.allclose(np.array([3,2,1,0]), snids)
        assert np.allclose(np.array([0,1,2,3]), dnids)
        assert np.allclose(np.concatenate([np.array([[0.2/1.0],[-0.3/1.0],[0.3/1.0],[-0.2/1.0]]),
                                           np.array([[1.2/1.4],[1.0],[1.3/1.4],[0.]])],
                                           axis=1),
                           feats)
        snids, dnids, feats = result[('n1','r1','n0')]['ef']
        assert np.allclose(np.array([4,5,6,7]), snids)
        assert np.allclose(np.array([4,5,6,7]), dnids)
        assert np.allclose(np.array([[1.1],[1.2],[-1.2],[0.9]]),
                           feats)

def test_node_label_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_label(Path(tmpdirname), 'node_label.csv')

        label_loader = data.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'node_label.csv'))
        label_loader.addTrainSet([0,1])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_nids)
        assert valid_nids is None
        assert test_nids is None
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert valid_labels is None
        assert test_labels is None
        label_loader.addValidSet([0,2])
        label_loader.addTestSet([0,3])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_nids)
        assert np.array_equal(np.array([0,1,2,3]), valid_nids)
        assert np.array_equal(np.array([0,1,2,3]), test_nids)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), valid_labels)
        assert np.array_equal(np.array([[0,0,1],[0,1,0],[1,0,0],[1,0,0]]), test_labels)

        # test with node type
        label_loader = data.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'node_label.csv'))
        label_loader.addTrainSet([0,1], node_type='n1')
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        label_loader.addValidSet([0,2], rows=[1,2,3], node_type='n1')
        label_loader.addTestSet([0,3], rows=[0,1,2], node_type='n1')
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert 'n1' in result
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result['n1']
        assert np.array_equal(np.array([3,2,1,0]), train_nids)
        assert np.array_equal(np.array([2,1,0]), valid_nids)
        assert np.array_equal(np.array([3,2,1]), test_nids)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert np.array_equal(np.array([[0,1,0],[0,0,1],[1,0,0]]), valid_labels)
        assert np.array_equal(np.array([[0,0,1],[0,1,0],[1,0,0]]), test_labels)

        # test multilabel
        # test with node type
        label_loader = data.NodeLabelLoader(os.path.join(tmpdirname,
                                                         'node_label.csv'))
        label_loader.addTrainSet(['node','label4'],
                                 multilabel=True,
                                 separator=',',
                                 node_type='n1')
        label_loader.addSet(['node', 'label5'],
                            split_rate=[0.,0.5,0.5],
                            multilabel=True,
                            separator=',',
                            node_type='n1')
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        np.random.seed(0)
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert 'n1' in result
        train_nids, train_labels, valid_nids, valid_labels, test_nids, test_labels = result['n1']
        label_map = label_loader.label_map
        rev_map = {val:key for key,val in label_map.items()}
        vl_truth = np.zeros((2,3),dtype='int32')
        vl_truth[0][rev_map['A']] = 1
        vl_truth[1][rev_map['A']] = 1
        vl_truth[1][rev_map['B']] = 1
        tl_truth = np.zeros((2,3),dtype='int32')
        tl_truth[0][rev_map['B']] = 1
        tl_truth[1][rev_map['A']] = 1
        tl_truth[1][rev_map['C']] = 1
        assert np.array_equal(np.array([3,2,1,0]), train_nids)
        assert np.array_equal(np.array([1,0]), valid_nids)
        assert np.array_equal(np.array([2,3]), test_nids)
        assert np.array_equal(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]), train_labels)
        assert np.array_equal(vl_truth, valid_labels)
        assert np.array_equal(tl_truth, test_labels)

def test_edge_label_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multiple_label(Path(tmpdirname), 'edge_label.csv')

        label_loader = data.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        # only existence of the edge
        label_loader.addTrainSet([0,6])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_snids)
        assert np.array_equal(np.array([2,3,4,5]), train_dnids)
        assert valid_snids is None
        assert valid_dnids is None
        assert test_snids is None
        assert test_dnids is None
        assert train_labels is None
        assert valid_labels is None
        assert test_labels is None
        label_loader.addValidSet([0,7])
        label_loader.addTestSet([6,8])
        node_dicts = {}
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[None]
        assert np.array_equal(np.array([0,1,2,3]), train_snids)
        assert np.array_equal(np.array([2,3,4,5]), train_dnids)
        assert np.array_equal(np.array([0,1,2,3]), valid_snids)
        assert np.array_equal(np.array([0,1,0,1]), valid_dnids)
        assert np.array_equal(np.array([2,3,4,5]), test_snids)
        assert np.array_equal(np.array([3,4,5,6]), test_dnids)

        # with labels
        label_loader = data.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        label_loader.addTrainSet([0,6,1], edge_type=('n1', 'like', 'n1'))
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        label_loader.addValidSet(['node', 'node_d2', 'label2'], rows=[1,2,3], edge_type=('n1', 'like', 'n1'))
        label_loader.addTestSet(['node_d', 'node_d3', 'label3'], rows=[0,1,2], edge_type=('n1', 'like', 'n1'))
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert ('n1', 'like', 'n1') in result
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('n1', 'like', 'n1')]
        assert np.array_equal(np.array([3,2,1,0]), train_snids)
        assert np.array_equal(np.array([1,0,4,5]), train_dnids)
        assert np.array_equal(np.array([2,1,0]), valid_snids)
        assert np.array_equal(np.array([2,3,2]), valid_dnids)
        assert np.array_equal(np.array([1,0,4]), test_snids)
        assert np.array_equal(np.array([0,4,5]), test_dnids)
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), train_labels)
        assert np.array_equal(np.array([[0,1,0],[0,0,1],[1,0,0]]), valid_labels)
        assert np.array_equal(np.array([[0,0,1],[0,1,0],[1,0,0]]), test_labels)

        # with multiple labels
        label_loader = data.EdgeLabelLoader(os.path.join(tmpdirname,
                                                         'edge_label.csv'))
        label_loader.addTrainSet(['node','node_d','label4'],
                                 multilabel=True,
                                 separator=',',
                                 edge_type=('n1', 'like', 'n2'))
        node_dicts = {'n1':{'node1':3,
                            'node2':2,
                            'node3':1,
                            'node4':0}}
        label_loader.addSet(['node_d2', 'node_d3', 'label5'],
                            split_rate=[0.,0.5,0.5],
                            multilabel=True,
                            separator=',',
                            edge_type=('n1', 'like', 'n2'))
        np.random.seed(0)
        result = label_loader.process(node_dicts)
        assert len(result) == 1
        assert ('n1', 'like', 'n2') in result
        train_snids, train_dnids, train_labels, \
            valid_snids, valid_dnids, valid_labels, \
            test_snids, test_dnids, test_labels = result[('n1', 'like', 'n2')]
        label_map = label_loader.label_map
        rev_map = {val:key for key,val in label_map.items()}
        vl_truth = np.zeros((2,3),dtype='int32')
        vl_truth[0][rev_map['A']] = 1
        vl_truth[1][rev_map['A']] = 1
        vl_truth[1][rev_map['B']] = 1
        tl_truth = np.zeros((2,3),dtype='int32')
        tl_truth[0][rev_map['B']] = 1
        tl_truth[1][rev_map['A']] = 1
        tl_truth[1][rev_map['C']] = 1
        assert np.array_equal(np.array([3,2,1,0]), train_snids)
        assert np.array_equal(np.array([0,1,2,3]), train_dnids)
        assert np.array_equal(np.array([3,2]), valid_snids)
        assert np.array_equal(np.array([3,4]), valid_dnids)
        assert np.array_equal(np.array([2,3]), test_snids)
        assert np.array_equal(np.array([2,1]), test_dnids)
        assert np.array_equal(np.array([[1,1,0],[1,0,0],[0,1,1],[1,0,1]]), train_labels)
        assert np.array_equal(vl_truth, valid_labels)
        assert np.array_equal(tl_truth, test_labels)

def test_edge_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'graphs.csv')

        edge_loader = data.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))

        edge_loader.addEdges([0,1])
        edge_loader.addEdges(['node_0','node_1'])
        edge_loader.addEdges(['node_0','node_1'],
                             rows=np.array([1,2,3,4]),
                             edge_type=('src', 'edge', 'src'))
        node_dicts = {}
        result = edge_loader.process(node_dicts)
        assert len(result) == 2
        snids, dnids = result[None]
        assert np.array_equal(np.array([0,1,2,3,3,0,1,2,3,3]), snids)
        assert np.array_equal(np.array([3,2,1,0,3,3,2,1,0,3]), dnids)
        snids, dnids = result[('src', 'edge', 'src')]
        assert np.array_equal(np.array([0,1,2,2]), snids)
        assert np.array_equal(np.array([1,0,3,2]), dnids)

        # with categorical relation
        edge_loader = data.EdgeLoader(os.path.join(tmpdirname,
                                                   'graphs.csv'))
        edge_loader.addCategoryRelationEdge([0,1,2],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_2'],
                                            src_type='src_t',
                                            dst_type='dst_t')
        edge_loader.addCategoryRelationEdge(['node_0','node_1','rel_1'],
                                            rows=np.array([1,2,3,4]),
                                            src_type='src',
                                            dst_type='dst')
        node_dicts = {'src_t':{'node1':3,
                                'node2':2,
                                'node3':1,
                                'node4':0}}
        result = edge_loader.process(node_dicts)
        assert len(result) == 4
        snids, dnids = result[('src_t','A','dst_t')]
        assert np.array_equal(np.array([3,2,1,0,0,0]), snids)
        assert np.array_equal(np.array([0,1,2,3,0,0]), dnids)
        snids, dnids = result[('src_t','B','dst_t')]
        assert np.array_equal(np.array([0]), snids)
        assert np.array_equal(np.array([3]), dnids)
        snids, dnids = result[('src_t','C','dst_t')]
        assert np.array_equal(np.array([3,2,1]), snids)
        assert np.array_equal(np.array([0,1,2]), dnids)
        snids, dnids = result[('src','A','dst')]
        assert np.array_equal(np.array([0,1,2,2]), snids)
        assert np.array_equal(np.array([0,1,2,3]), dnids)


def test_build_graph():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_graph_edges(Path(tmpdirname), 'edges.csv')
        create_edge_labels(Path(tmpdirname), 'edge_labels.csv')
        create_node_labels(Path(tmpdirname), 'node_labels.csv')

        # homogeneous graph loader
        node_feat_loader = data.NodeFeatureLoader(os.path.join(tmpdirname, 'node_labels.csv'))
        node_feat_loader.addCategoryFeature([0,1])
        node_feat_loader.addMultiCategoryFeature([0,2], separator=',')
        edge_label_loader = data.EdgeLabelLoader(os.path.join(tmpdirname, 'edge_labels.csv'))
        edge_label_loader.addSet([0,1,2],split_rate=[0.5,0.25,0.25])
        edge_loader = data.EdgeLoader(os.path.join(tmpdirname, 'edges.csv'))
        edge_loader.addEdges([0,1])

        np.random.seed(0)
        graphloader = data.GraphLoader(name='example')
        graphloader.appendEdge(edge_loader)
        graphloader.appendLabel(edge_label_loader)
        graphloader.appendFeature(node_feat_loader)
        graphloader.process()

        node_id_map = graphloader.node_2_id
        assert None in node_id_map
        assert len(node_id_map[None]) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert node_id_map[None][key] == idx
        id_node_map = graphloader.id_2_node
        assert None in id_node_map
        assert len(id_node_map[None]) == 4
        for idx, key in enumerate(['node1', 'node2', 'node3', 'node4']):
            assert id_node_map[None][idx] == key
        label_map = graphloader.label_map
        assert len(label_map) == 2
        assert label_map[0] == 'A'
        assert label_map[1] == 'C'

        g = graphloader.graph
        assert g.num_edges() == 9
        assert np.array_equal(g.edata['labels'].long().numpy(),
            np.array([[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[0,1],[1,0],[1,0],[1,0]]))
        assert th.nonzero(g.edata['train_mask']).shape[0] == 2
        assert th.nonzero(g.edata['valid_mask']).shape[0] == 1
        assert th.nonzero(g.edata['test_mask']).shape[0] == 1
        assert np.allclose(g.ndata['nf'].numpy(),
            np.array([[1,0,1,0,0,1,0,0,0],[1,0,0,0,1,1,1,0,0],[0,1,1,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,1]]))


if __name__ == '__main__':
    #test_minigc()
    #test_data_hash()

    #test_row_normalize()
    #test_col_normalize()
    #test_float_row_normalize()
    #test_float_col_normalize()
    #test_float_col_maxmin_normalize()
    #test_embed_word2vec()

    #test_parse_lang_feat()
    #test_parse_category_feat()
    #test_parse_numerical_feat()
    #test_parse_numerical_multihot_feat()

    # test Feature Loader
    #test_node_category_feature_loader()
    #test_node_numerical_feature_loader()
    #test_node_word2vec_feature_loader()
    #test_edge_numerical_feature_loader()
    # test Label Loader
    #test_node_label_loader()
    #test_edge_label_loader()
    # test Edge Loader
    test_edge_loader()

    # test feature process
    #test_node_feature_process()
    #test_edge_feature_process()
    # test label process
    #test_node_label_process()
    #test_edge_label_process()
    # test edge process
    #test_edge_process()

    test_build_graph()