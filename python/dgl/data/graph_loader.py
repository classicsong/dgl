""" Classes for loading raw graph"""
import os
import csv

import numpy as np
# TODO(xiangsx): Framework agnostic later
import torch as th
import dgl

from .utils import save_graphs, load_graphs, save_info, load_info
from .utils import field2idx, get_id
from .feature_loader import NodeFeatureLoader, EdgeFeatureLoader
from .label_loader import NodeLabelLoader, EdgeLabelLoader

class EdgeLoader(object):
    r"""EdgeLoader allows users to define graph edges.

    Parameters
    ----------
    input: str
        Data source, for the csv file input,
        it should be a string of file path
    separator: str
        Delimiter(separator) used in csv file.
        Default: '\t'
    has_header: bool
        Whether the input data has col name.
        Default: False
    int_id: bool
        Whether the raw node id is an int,
        this can help speed things up.
        Default: False
    eager_mode: bool
        Whether to use eager parse mode.
        See **Note** for more details.
        Default: False
    verbose: bool, optional
        Whether print debug info during parsing
        Default: False

    Note:

    * Currently, we only support raw csv file input.

    * If eager_mode is True, the loader will processing
    the edges immediately after addXXXEdges
    is called. This will case extra performance overhead
    when merging multiple edge loaders together to
    build the DGLGraph.

    * If eager_mode if False, the edges are not
    processed until building the DGLGraph.

    Examples:

    ** Creat a FeatureLoader to load user features from u.csv.**

    >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                             separator="|")
    >>> user_loader.addCategoryFeature(cols=["id", "gender"], node_type='user')

    ** create node label loader to load labels **

    >>> label_loader = dgl.data.NodeLabelLoader(input='label.csv',
                                                separator="|")
    >>> label_loader.addTrainSet([0, 1], rows=np.arange(start=0,
                                                        stop=100))

    ** create edge loader to load edges **

    >>> edge_loader = dgl.data.EdgeLoader(input='edge.csv',
                                          separator="|")
    >>> edge_loader.addEdges([0, 1])

    ** Append features into graph loader **
    >>> graphloader = dgl.data.GraphLoader()
    >>> graphloader.appendEdge(edge_loader)
    >>> graphloader.appendFeature(user_loader)
    >>> graphloader.appendLabel(label_loader)

    """
    def __init__(self, input, separator='\t', has_head=True, int_id=False,
        eager_mode=False, encoding='utf-8', verbose=False):
        if not os.path.exists(input):
            raise RuntimeError("File not exist {}".format(input))

        assert eager_mode is False, "Currently we do not support eager_mode"

        self._input = input
        self._separator = separator
        self._has_head = has_head
        self._int_id = int_id
        self._eager_mode = eager_mode
        self._encoding = encoding
        self._verbose = verbose
        self._edges = []

    def process(self, node_dicts):
        """ preparing edges for creating dgl graph.

        Src and dst nodes are converted into consecutive integer ID spaces

        Params:
        node_dicts: dict of dict
            {node_type: {node_str : node_id}}

        Return:
            dict
            {edge_type: (snids, dnids)}
        """
        results = {}
        for edges in self._edges:
            edge_type, src_nodes, dst_nodes = edges
            if edge_type is None:
                src_type = None
                dst_type = None
            else:
                src_type, rel_type, dst_type = edge_type

            # convert src node and dst node
            if src_type in node_dicts:
                snid_map = node_dicts[src_type]
            else:
                snid_map = {}
                node_dicts[src_type] = snid_map

            if dst_type in node_dicts:
                dnid_map = node_dicts[dst_type]
            else:
                dnid_map = {}
                node_dicts[dst_type] = dnid_map

            snids = []
            dnids = []
            for node in src_nodes:
                nid = get_id(snid_map, node)
                snids.append(nid)
            for node in dst_nodes:
                nid = get_id(dnid_map, node)
                dnids.append(nid)
            snids = np.asarray(snids, dtype='long')
            dnids = np.asarray(dnids, dtype='long')

            # chech if same node_type already exists
            # if so concatenate the edges.
            if edge_type in results:
                last_snids, last_dnids = results[edge_type]
                results[edge_type] = (np.concatenate((last_snids, snids)),
                                     np.concatenate((last_dnids, dnids)))
            else:
                results[edge_type] = (snids, dnids)
        return results

    def addEdges(self, cols, rows=None, edge_type=None):
        r""" Add edges into the graph

        Two columns of **input** are chosen, one for
        source node name and another for destination node name.
        Each row represetns an edge.

        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for nodes
            The first column is treated as source node name,
            the second column is treated as destination node name.
            (2) [int, int, int] column numbers nodes.
            The first column is treated as source node name,
            the second column is treated as destination node name.

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        edge_type: str
            Edge type. If None, default edge type is choose. Otherwise a tuple of
            (src_type, relation_type, dst_type) should be provided.
            Default: None

        Example:

        ** Load Edges **

        Example data of data.csv is as follows:

        ===  ===
        src  dst
        ===  ===
        1    0
        0    2
        3    4
        ===  ===

        >>> edgeloader = dgl.data.EdgeLoader()
        >>> edgeloader.addEdges(cols=["src", "dst"], edge_type=('user', 'like', 'movie'))

        """
        if not isinstance(cols, list):
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise RuntimeError("addEdges only accepts two columns "\
                           "for source node and destination node.")

        if edge_type != None and len(edge_type) != 3:
            raise RuntimeError("edge_type should be None or a tuple of " \
                "(src_type, relation_type, dst_type)")

        src_nodes = []
        dst_nodes = []
        labels = []
        with open(self._input, newline='', encoding=self._encoding) as csvfile:
            if isinstance(cols[0], str):
                assert self._has_head, \
                    "The column name is provided to identify the target column." \
                    "The input csv should have the head field"
                reader = csv.reader(csvfile, delimiter=self._separator)
                heads = next(reader)
                # find index of each target field name
                idx_cols = field2idx(cols, heads)

                assert len(idx_cols) == len(cols), \
                    "one or more field names are not found in {}".format(self._input)
                cols = idx_cols
            else:
                reader = csv.reader(csvfile, delimiter=self._separator)
                if self._has_head:
                    # skip field name
                    next(reader)

            # fast path, all rows are used
            if rows is None:
                for line in reader:
                    src_nodes.append(line[cols[0]])
                    dst_nodes.append(line[cols[1]])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        src_nodes.append(line[cols[0]])
                        dst_nodes.append(line[cols[1]])
                        row_idx += 1
                    # else skip this line

        self._edges.append((edge_type, src_nodes, dst_nodes))

    def addCategoryRelationEdge(self, cols, src_type, dst_type, rows=None):
        r"""Convert categorical features as edge relations

        Three columns of the **input** are chosen, one for
        source node name, one for destination node name
        and the last for category type. The triple
        (src_name, category_type, dst_name) is treated as
        canonical edge type.
        Each row represetns an edge.

        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str, str] column names for node and categorical data.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as edge category.
            (2) [int, int, int] column numbers for node and categorical data.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as edge category.

        src_type: str
            Source node type.

        dst_type: str
            Destination node type.

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        Example:

        ** Load Category Relation **

        Example data of data.csv is as follows:

        ====    ====    ========
        name    rate    movie
        ====    ====    ========
        John    5       StarWar1
        Tim     3.5     X-Man
        Maggy   4.5     StarWar1
        ====    ====    ========

        >>> edgeloader = dgl.data.EdgeLoader()
        >>> edgeloader.addCategoryRelationEdge(cols=["name", "rate", "movie"], src_type='user', dst_type='movie')

        """
        if not isinstance(cols, list):
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) != 3:
            raise RuntimeError("addCategoryRelationEdge only accepts three columns " \
                           "the first column for source node, " \
                           "the second for destination node, " \
                           "and third for edge category")

        edges = {}
        labels = []
        with open(self._input, newline='', encoding=self._encoding) as csvfile:
            if isinstance(cols[0], str):
                assert self._has_head, \
                    "The column name is provided to identify the target column." \
                    "The input csv should have the head field"
                reader = csv.reader(csvfile, delimiter=self._separator)
                heads = next(reader)
                # find index of each target field name
                idx_cols = field2idx(cols, heads)

                assert len(idx_cols) == len(cols), \
                    "one or more field names are not found in {}".format(self._input)
                cols = idx_cols
            else:
                reader = csv.reader(csvfile, delimiter=self._separator)
                if self._has_head:
                    # skip field name
                    next(reader)

            # fast path, all rows are used
            if rows is None:
                for line in reader:
                    rel_type = line[cols[2]]
                    if rel_type in edges:
                        edges[rel_type][0].append(line[cols[0]])
                        edges[rel_type][1].append(line[cols[1]])
                    else:
                        edges[rel_type] = ([line[cols[0]]], [line[cols[1]]])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        rel_type = line[cols[2]]
                        if rel_type in edges:
                            edges[rel_type][0].append(line[cols[0]])
                            edges[rel_type][1].append(line[cols[1]])
                        else:
                            edges[rel_type] = ([line[cols[0]]], [line[cols[1]]])
                        row_idx += 1
                    # else skip this line
        for key, val in edges.items():
            self._edges.append(((src_type, key, dst_type), val[0], val[1]))

class GraphLoader(object):
    r""" Generate DGLGraph by parsing files.

    GraphLoader generate DGLGraph by collecting EdgeLoaders,
    FeatureLoaders and LabelLoders and parse them iteratively.

    Parameters
    ----------
    name: str, optional
        name of the graph.
        default: 'graph'
    edge_loader: list of EdgeLoader
        edge loaders to load graph edges
        default: None
    feature_loader: list of NodeFeatureLoader and EdgeFeatureLoader
        feature loaders to load graph features and edges
        default: None
    label_loader: list of NodeLabelLoader and EdgeLabelLoader
        label loaders to load labels/ground-truth and edges
        default: None
    verbose: bool, optional
        Whether print debug info during parsing
        Default: False

    Note:
    -----

    **EdgeLoader is used to add edges that neither have features nor appeared in
    edge labels.** If one edge appears in both appendEdge and appendLabel, it will
    be added twice. **But if one edge appears in both EdgeFeatureLoader and
    EdgeLabelLoader, it will only be added once.


    Example:

    ** Create a Graph Loader **

    >>> graphloader = dgl.data.GraphLoader(name='example')

    """
    def __init__(self, name='graph',
        edge_loader=None, feature_loader=None, label_loader=None, verbose=False):
        self._name = name

        if edge_loader is not None:
            if not isinstance(edge_loader, list):
                raise RuntimeError("edge loaders should be a list of EdgeLoader")

            self._edge_loader = edge_loader
        else:
            self._edge_loader = []

        if feature_loader is not None:
            if not isinstance(feature_loader, list):
                raise RuntimeError("feature loaders should be " \
                    "a list of NodeFeatureLoader and EdgeFeatureLoader")

            self._feature_loader = feature_loader
        else:
            self._feature_loader = []

        if label_loader is not None:
            if not isinstance(label_loader, list):
                raise RuntimeError("label loaders should be " \
                    "a list of NodeLabelLoader and EdgeLabelLoader")

            self._label_loader = label_loader
        else:
            self._label_loader = []

        self._graph = None
        self._verbose = verbose
        self._node_dict = {}
        self._label_map = None

    def appendEdge(self, edge_loader):
        """ Add edges into graph

        Parameters
        ----------
        edge_loader: EdgeLoader
            edge loaders to load graph edges
            default: None

        Example:

        ** create edge loader to load edges **

        >>> edge_loader = dgl.data.EdgeLoader(input='edge.csv',
                                            separator="|")
        >>> edge_loader.addEdges([0, 1])

        ** Append edges into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendEdge(edge_loader)

        """
        if not isinstance(edge_loader, EdgeLoader):
            raise RuntimeError("edge loader should be a EdgeLoader")
        self._edge_loader.append(edge_loader)

    def appendFeature(self, feature_loader):
        """ Add features and edges into graph

        Parameters
        ----------
        feature_loader: NodeFeatureLoader or EdgeFeatureLoader
            feature loaders to load graph edges
            default: None

        Example:

        ** Creat a FeatureLoader to load user features from u.csv.**

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|")
        >>> user_loader.addCategoryFeature(cols=["id", "gender"], node_type='user')

        ** Append features into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendFeature(user_loader)

        """
        if not isinstance(feature_loader, NodeFeatureLoader) and \
            not isinstance(feature_loader, EdgeFeatureLoader):
            raise RuntimeError("feature loader should be a NodeFeatureLoader or EdgeFeatureLoader.")
        self._feature_loader.append(feature_loader)

    def appendLabel(self, label_loader):
        """ Add labels and edges into graph

        Parameters
        ----------
        label_loader: NodeLabelLoader or EdgeLabelLoader
            label loaders to load graph edges
            default: None

        Note
        ----
        To keep the overall design of the GraphLoader simple,
        it accepts only one LabelLoader.

        Examples:

        ** create node label loader to load labels **

        >>> label_loader = dgl.data.NodeLabelLoader(input='label.csv',
                                                    separator="|")
        >>> label_loader.addTrainSet([0, 1], rows=np.arange(start=0,
                                                            stop=100))

        ** Append labels into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendLabel(label_loader)

        """
        if not isinstance(label_loader, NodeLabelLoader) and \
            not isinstance(label_loader, EdgeLabelLoader):
            raise RuntimeError("label loader should be a NodeLabelLoader or EdgeLabelLoader.")
        assert len(self._label_loader) == 0, \
            'DGL GraphLoader only support one label loader now.' \
            'It requires no extra efforts to sync the label mappings'
        self._label_loader.append(label_loader)

    def addReverseEdge(self):
        """ Add Reverse edges with new relation type.

        addReverseEdge works for heterogenous graphs. It adds
        a new relation type for each existing relation. For
        example, with relation ('head', 'rel', 'tail'), it will
        create a new relation type ('tail', 'rev-rel', 'head')
        and adds edges belong to ('head', 'rel', 'tail') into
        new relation type with reversed head and tail entity order.

        Example:

        ** create edge loader to load edges **

        >>> edge_loader = dgl.data.EdgeLoader(input='edge.csv',
                                            separator="|")
        >>> edge_loader.addEdges([0, 1],
                                src_type='user',
                                edge_type='likes',
                                dst_type='movie')

        ** Append edges into graph loader **

        >>> graphloader = dgl.data.GraphLoader(name='example')
        >>> graphloader.appendEdge(edge_loader)

        ** add reversed edges into graph **

        >>> graphloader.addReverseEdge()

        """
        assert self._g.is_homogeneous is False, \
            'Add reversed edges only work for heterogeneous graph'

        new_g = dgl.add_reverse_edges(self._g, copy_ndata=True, copy_edata=False)
        for etype in self._g.canonical_etypes:
            new_g.edges[etype].data = self._g.edges[etype].data
        self._g = new_g

    def process(self):
        """ Parsing EdgeLoaders, FeatureLoaders and LabelLoders to build the DGLGraph
        """
        graphs = {} # edge_type: (s, d, feat)
        nodes = {}
        edge_feat_results = []
        node_feat_results = []
        edge_label_results = []
        node_label_results = []

        if self._verbose:
            print('Start processing graph structure ...')
        # we first handle edges
        for edge_loader in self._edge_loader:
            # {edge_type: (snids, dnids)}
            edge_result = edge_loader.process(self._node_dict)
            for edge_type, vals in edge_result.items():
                snids, dnids = vals
                if edge_type in graphs:
                    graphs[edge_type] = (np.concatenate((graphs[edge_type][0], snids)),
                                         np.concatenate((graphs[edge_type][1], dnids)),
                                         None)
                else:
                    graphs[edge_type] = (snids, dnids, None)

        # we assume edges have features is not loaded by edgeLoader.
        for feat_loader in self._feature_loader:
            if feat_loader.node_feat is False:
                # {edge_type: (snids, dnids, feats)}
                edge_feat_result = feat_loader.process(self._node_dict)
                edge_feat_results.append(edge_feat_result)

                for edge_type, vals in edge_feat_result.items():
                    if edge_type in graphs:
                        snids, dnids, feats = vals
                        new_feats = {}
                        for feat_name, feat in feats:
                            assert feat_name not in graphs[edge_type][2], \
                                'Can not concatenate edges with features with other edges without features'
                            assert graphs[edge_type][2][feat_name].shape[1:] == feat.shape[1:], \
                                'Can not concatenate edges with different feature shape'

                            new_feats[feat_name] = np.concatenate((graphs[edge_type][2][feat_name], feat))

                        graphs[edge_type] = (np.concatenate((graphs[edge_type][0], snids)),
                                             np.concatenate((graphs[edge_type][1], dnids)),
                                             new_feats)
                    else:
                        graphs[edge_type] = vals
            else:
                # {node_type: {feat_name :(node_ids, node_feats)}}
                node_feat_result = feat_loader.process(self._node_dict)
                node_feat_results.append(node_feat_result)

                for node_type, vals in node_feat_result.items():
                    nids, _ = vals[next(iter(vals.keys()))]
                    max_nid = int(np.max(nids)) + 1
                    if node_type in nodes:
                        nodes[node_type] = max(nodes[node_type]+1, max_nid)
                    else:
                        nodes[node_type] = max_nid

        assert len(self._label_loader) == 1, \
            'DGL GraphLoader should only have one label loader that ' \
            'it requires no extra efforts to sync the label mappings'

        # TODO(xiangsx): in future we may support multiple LabelLoaders.
        for label_loader in self._label_loader:
            self._label_map = label_loader.label_map
            if label_loader.node_label is False:
                # {edge_type: ((train_snids, train_dnids, train_labels,
                #               valid_snids, valid_dnids, valid_labels,
                #               test_snids, test_dnids, test_labels)}
                edge_label_result = label_loader.process(self._node_dict)
                edge_label_results.append(edge_label_result)

                for edge_type, vals in edge_label_result.items():
                    train_snids, train_dnids, train_labels, \
                        valid_snids, valid_dnids, valid_labels, \
                        test_snids, test_dnids, test_labels = vals

                    if edge_type in graphs:
                        # If same edge_type also has features,
                        # we expect edges have labels also have features.
                        # Thus we avoid add edges twice.
                        # Otherwise, if certain edge_type has no featus, add it directly
                        if graphs[edge_type][2] is None:
                            snids = graphs[edge_type][0]
                            dnids = graphs[edge_type][1]
                            if train_snids is not None:
                                snids = np.concatenate((snids, train_snids))
                                dnids = np.concatenate((dnids, train_dnids))
                            if valid_snids is not None:
                                snids = np.concatenate((snids, valid_snids))
                                dnids = np.concatenate((dnids, valid_dnids))
                            if test_snids is not None:
                                snids = np.concatenate((snids, test_snids))
                                dnids = np.concatenate((dnids, test_dnids))
                            graphs[edge_type] = (snids, dnids, None)
                    else:
                        snids = np.empty((0,), dtype='long')
                        dnids = np.empty((0,), dtype='long')
                        if train_snids is not None:
                            snids = np.concatenate((snids, train_snids))
                            dnids = np.concatenate((dnids, train_dnids))
                        if valid_snids is not None:
                            snids = np.concatenate((snids, valid_snids))
                            dnids = np.concatenate((dnids, valid_dnids))
                        if test_snids is not None:
                            snids = np.concatenate((snids, test_snids))
                            dnids = np.concatenate((dnids, test_dnids))
                        graphs[edge_type] = (snids, dnids, None)
            else:
                # {node_type: (train_nids, train_labels,
                #              valid_nids, valid_labels,
                #              test_nids, test_labels)}
                node_label_result = label_loader.process(self._node_dict)
                node_label_results.append(node_label_result)
                for node_type, vals in node_label_result.items():
                    train_nids, _, valid_nids, _, test_nids, _ = vals
                    max_nid = 0
                    if train_nids is not None:
                        max_nid = max(int(np.max(train_nids), max_nid))
                    if valid_nids is not None:
                        max_nid = max(int(np.max(valid_nids), max_nid))
                    if test_nids is not None:
                        max_nid = max(int(np.max(test_nids), max_nid))

                    if node_type in nodes:
                        nodes[node_type] = max(nodes[node_type], max_nid)
                    else:
                        nodes[node_type] = max_nid
        if self._verbose:
            print('Done processing graph structure.')
            print('Start building dgl graph.')

        # build graph
        if len(graphs) > 1:
            assert None not in graphs, \
                'With heterogeneous graph, all edges should have edge type'
            assert None not in nodes, \
                'With heterogeneous graph, all nodes should have node type'
            graph_edges = {key: (val[0], val[1]) for key, val in graphs.items()}
            g = dgl.heterograph(graph_edges, num_nodes=nodes)
            for edge_type, vals in graphs.items():
                # has edge features
                if vals[2] is not None:
                    for key, feat in vals[2].items():
                        g.edges[edge_type].data[key] = th.tensor(feat)
        else:
            g = dgl.graph((graphs[None][0], graphs[None][1]), num_nodes=nodes[None])
            # has edge features
            if graphs[None][2] is not None:
                for key, feat in graphs[None][2].items():
                    g.edata[key] = th.tensor(feat)

        # no need to handle edge features
        # handle node features
        for node_feats in node_feat_results:
            # {node_type: (node_ids, node_feats)}
            for node_type, vals in node_feats.items():
                if node_type is None:
                    for key, feat in vals.items():
                        g.ndata[key] = th.tensor(feat[1])
                else:
                    for key, feat in vals.items():
                        g.nodes[node_type].data[key] = th.tensor(feat[1])

        if self._verbose:
            print('Done building dgl graph.')
            print('Start processing graph labels...')
        train_edge_labels = {}
        valid_edge_labels = {}
        test_edge_labels = {}
        # concatenate all edge labels
        for edge_label_result in edge_label_results:
            for edge_type, vals in edge_label_result.items():
                train_snids, train_dnids, train_labels, \
                    valid_snids, valid_dnids, valid_labels, \
                    test_snids, test_dnids, test_labels = vals

                # train edge labels
                if train_snids is not None:
                    if edge_type in train_edge_labels:
                        train_edge_labels[edge_type] = (
                            np.concatenate((train_edge_labels[edge_type][0], train_snids)),
                            np.concatenate((train_edge_labels[edge_type][1], train_dnids)),
                            None if train_labels is None else \
                                np.concatenate((train_edge_labels[edge_type][2], train_labels)))
                    else:
                        train_edge_labels[edge_type] = (train_snids, train_dnids, train_labels)

                # valid edge labels
                if valid_snids is not None:
                    if edge_type in valid_edge_labels:
                        valid_edge_labels[edge_type] = (
                            np.concatenate((valid_edge_labels[edge_type][0], valid_snids)),
                            np.concatenate((valid_edge_labels[edge_type][1], valid_dnids)),
                            None if valid_labels is None else \
                                np.concatenate((valid_edge_labels[edge_type][2], valid_labels)))
                    else:
                        valid_edge_labels[edge_type] = (valid_snids, valid_dnids, valid_labels)

                # test edge labels
                if test_snids is not None:
                    if edge_type in test_edge_labels:
                        test_edge_labels[edge_type] = (
                            np.concatenate((test_edge_labels[edge_type][0], test_snids)),
                            np.concatenate((test_edge_labels[edge_type][1], test_dnids)),
                            None if test_labels is None else \
                                np.concatenate((test_edge_labels[edge_type][2], test_labels)))
                    else:
                        test_edge_labels[edge_type] = (test_snids, test_dnids, test_labels)

        # create labels and train/valid/test mask
        assert len(train_edge_labels) >= len(valid_edge_labels), \
            'The training set should cover all kinds of edge types ' \
            'where the validation set is avaliable.'
        assert len(train_edge_labels) == len(test_edge_labels), \
            'The training set should cover the same edge types as the test set.'

        for edge_type, train_val in train_edge_labels.items():
            train_snids, train_dnids, train_labels = train_val
            if edge_type in valid_edge_labels:
                valid_snids, valid_dnids, valid_labels = valid_edge_labels[edge_type]
            else:
                valid_snids, valid_dnids, valid_labels = None, None, None
            assert edge_type in test_edge_labels
            test_snids, test_dnids, test_labels = test_edge_labels[edge_type]

            u, v, eids = g.edge_ids(train_snids,
                                    train_dnids,
                                    return_uv=True,
                                    etype=edge_type)
            labels = None
            # handle train label
            if train_labels is not None:
                assert train_snids.shape[0] == eids.shape[0], \
                    'Under edge type {}, There exists multiple edges' \
                    'between some (src, dst) pair in the training set.' \
                    'This is misleading and will not be supported'.format(
                        edge_type if edge_type is not None else "")
                train_labels = th.tensor(train_labels)
                labels = th.full((g.num_edges(edge_type), train_labels.shape[1]),
                                    -1,
                                    dtype=train_labels.dtype)
                labels[eids] = train_labels
            # handle train mask
            train_mask = th.full((g.num_edges(edge_type),), False)
            train_mask[eids] = True

            valid_mask = None
            if valid_snids is not None:
                u, v, eids = g.edge_ids(valid_snids,
                                        valid_dnids,
                                        return_uv=True,
                                        etype=edge_type)
                assert valid_snids.shape[0] == eids.shape[0], \
                    'Under edge type {}, There exists multiple edges' \
                    'between some (src, dst) pair in the validation set.' \
                    'This is misleading and will not be supported'.format(
                        edge_type if edge_type is not None else "")
                # handle valid label
                if valid_labels is not None:
                    assert labels is not None, \
                        'We must have train_labels first then valid_labels'
                    labels[eids] = th.tensor(valid_labels)
                # handle valid mask
                valid_mask = th.full((g.num_edges(edge_type),), False)
                valid_mask[eids] = True

            u, v, eids = g.edge_ids(test_snids,
                                    test_dnids,
                                    return_uv=True,
                                    etype=edge_type)
            # handle test label
            if test_labels is not None:
                assert test_snids.shape[0] == eids.shape[0], \
                    'Under edge type {}, There exists multiple edges' \
                    'between some (src, dst) pair in the testing set.' \
                    'This is misleading and will not be supported'.format(
                        edge_type if edge_type is not None else "")
                assert labels is not None, \
                    'We must have train_labels first then test_lavbels'
                labels[eids] = th.tensor(test_labels)
            # handle test mask
            test_mask = th.full((g.num_edges(edge_type),), False)
            test_mask[eids] = True

            # add label and train/valid/test masks into g
            if edge_type is None:
                assert len(train_edge_labels) == 1, \
                    'Homogeneous graph only supports one type of labels'
                g.edata['labels'] = labels
                g.edata['train_mask'] = train_mask
                g.edata['valid_mask'] = valid_mask
                g.edata['test_mask'] = test_mask
            else: # we have edge type
                assert 'train_mask' not in g.edges[edge_type].data
                g.edges[edge_type].data['labels'] = labels
                g.edges[edge_type].data['train_mask'] = train_mask
                g.edges[edge_type].data['valid_mask'] = valid_mask
                g,edges[edge_type].data['test_mask'] = test_mask

        # node labels
        train_node_labels = {}
        valid_node_labels = {}
        test_node_labels = {}
        for node_labels in node_label_results:
            for node_type, vals in node_label_result.items():
                train_nids, train_labels, \
                    valid_nids, valid_labels, \
                    test_nids, test_labels = vals

                # train node labels
                if node_type in train_node_labels:
                    if train_nids is not None:
                        train_node_labels[node_type] = (
                            np.concatenate((train_node_labels[node_type][0], train_nids)),
                            None if train_labels is None else \
                                np.concatenate((train_node_labels[node_type][1], train_labels)))
                    else:
                        train_node_labels[node_type] = (train_nids, train_labels)

                # valid node labels
                if node_type in valid_node_labels:
                    if valid_nids is not None:
                        valid_node_labels[node_type] = (
                            np.concatenate((valid_node_labels[node_type][0], valid_nids)),
                            None if valid_labels is None else \
                                np.concatenate((valid_node_labels[node_type][0], valid_labels)))
                    else:
                        valid_node_labels[node_type] = (valid_nids, valid_labels)

                # test node labels
                if node_type in test_node_labels:
                    if test_nids is not None:
                        test_node_labels[node_type] = (
                            np.concatenate((test_node_labels[node_type][0], test_nids)),
                            None if test_labels is none else \
                                np.concatenate((test_node_labels[node_type][0], test_labels)))
                    else:
                        test_node_labels[node_type] = (test_nids, test_labels)

        # create labels and train/valid/test mask
        assert len(train_node_labels) >= len(valid_node_labels), \
            'The training set should cover all kinds of node types ' \
            'where the validation set is avaliable.'
        assert len(train_node_labels) == len(test_node_labels), \
            'The training set should cover the same node types as the test set.'

        for node_type, train_val in train_node_labels:
            train_nids, train_labels = train_val
            valid_nids, valid_labels = valid_node_labels[node_type] \
                if node_type in valid_node_labels else None, None
            test_nids, test_labels = test_node_labels[node_type]

            labels = None
            # handle train label
            if train_labels is not None:
                train_labels = th.tensor(train_labels)
                labels = th.full((g.num_nodes(node_type), train_labels.shape[1]),
                                 value=-1,
                                 dtype=train_labels.dtype)
                labels[train_nids] = train_labels
            # handle train mask
            train_mask = th.full((g.num_nodes(node_type),), False)
            train_mask[train_nids] = True

            valid_mask = None
            if valid_nids is not None:
                # handle valid label
                if valid_labels is not None:
                    assert labels is not None, \
                        'We must have train_labels first then valid_labels'
                    labels[valid_nids] = valid_labels
                # handle valid mask
                valid_mask = th.full((g.num_nodes(node_type),), False)
                valid_mask[valid_nids] = True

            # handle test label
            if test_labels is not None:
                assert labels is not None, \
                    'We must have train_labels first then test_labels'
                labels[test_nids] = test_labels
            # handle test mask
            test_mask = th.full((g.num_nodes(node_type),), False)
            test_mask[test_nids] = True

            # add label and train/valid/test masks into g
            if node_type is None:
                assert len(train_node_labels) == 1, \
                    'Homogeneous graph only supports one type of labels'
                g.ndata['labels'] = labels
                g.ndata['train_mask'] = train_mask
                g.ndata['valid_mask'] = valid_mask
                g.ndata['test_mask'] = test_mask
            else: # we have node type
                assert 'train_mask' not in g.nodes[node_type].data
                g.nodes[node_type].data['labels'] = labels
                g.nodes[node_type].data['train_mask'] = train_mask
                g.nodes[node_type].data['valid_mask'] = valid_mask
                g.nodes[node_type].data['test_mask'] = test_mask

        if self._verbose:
            print('Done processing labels')

        self._g = g

    def save(self, path):
        """save the graph and the labels"""
        graph_path = os.path.join(path,
                                  'graph.bin')
        info_path = os.path.join(path,
                                 'info.pkl')
        save_graphs(graph_path, self._g)
        save_info(info_path, {'node_id_map': self._node_dict,
                              'label_map': self._label_map})

    def load(self, path):
        graph_path = os.path.join(path,
                                  'graph.bin')
        info_path = os.path.join(path,
                                 'info.pkl')
        graphs, _ = load_graphs(graph_path)
        self._g = graphs[0]
        info = load_info(str(info_path))
        self._node_dict = info['node_id_map']
        self._label_map = info['label_map']

    @property
    def node_2_id(self):
        """ Return mappings from raw node id/name to internal node id

        Return
        ------
        dict of dict:
            {node_type : {raw node id(string/int): dgl_id}}
        """
        return self._node_dict

    @property
    def id_2_node(self):
        """ Return mappings from internal node id to raw node id/name

        Return
        ------
        dict of dict:
            {node_type : {raw node id(string/int): dgl_id}}
        """
        return {node_type : {val:key for key, val in node_maps.items()} \
            for node_type, node_maps in self._node_dict.items()}


    @property
    def label_map(self):
        """ Return mapping from internal label id to original label

        Return
        ------
        dict:
            {label id(int) : raw label(string/int)}
        """
        return self._label_map

    @property
    def graph(self):
        """ Return processed graph
        """
        return self._g