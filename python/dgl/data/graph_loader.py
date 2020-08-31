""" Classes for loading raw graph"""
import os
import csv

import numpy as np

from ..base import DGLError, dgl_warning
from .utils import field2idx, get_id

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
            snids = np.asarray(snids)
            dnids = np.asarray(dnids)

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
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addEdges only accepts two columns "\
                           "for source node and destination node.")

        if edge_type != None and len(edge_type) != 3:
            raise DGLError("edge_type should be None or a tuple of " \
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
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 3:
            raise DGLError("addCategoryRelationEdge only accepts three columns " \
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

    Example:

    ** Create a Graph Loader **

    >>> graphloader = dgl.data.GraphLoader(name='example')

    """
    def __init__(self, name='graph',
        edge_loader=None, feature_loader=None, label_loader=None, verbose=False):
        self._name = name

        if edge_loader is not None:
            if not isinstance(edge_loader, list):
                raise DGLError("edge loaders should be a list of EdgeLoader")

            self._edge_loader = edge_loader
        else:
            self._edge_loader = []

        if feature_loader is not None:
            if not isinstance(feature_loader, list):
                raise DGLError("feature loaders should be " \
                    "a list of NodeFeatureLoader and EdgeFeatureLoader")

            self._feature_loader = feature_loader
        else:
            self._feature_loader = []

        if label_loader is not None:
            if not isinstance(label_loader, list):
                raise DGLError("label loaders should be " \
                    "a list of NodeLabelLoader and EdgeLabelLoader")

            self._label_loader = label_loader
        else:
            self._label_loader = []

        self._graph = None
        self._verbose = verbose
        self._node_dict = {}

    def appendEdge(self, edge_loader):
        if not isinstance(edge_loader, EdgeLoader):
            raise DGLError("edge loader should be a EdgeLoader")
        self._edge_loader.append(edge_loader)

    def appendFeature(self, feature_loader):
        if not isinstance(feature_loader, NodeFeatureLoader) and \
            not isinstance(feature_loader, EdgeFeatureLoader):
            raise DGLError("feature loader should be a NodeFeatureLoader or EdgeFeatureLoader.")
        self._feature_loader.append(feature_loader)

    def appendLabel(self, label_loader):
        if not isinstance(label_loader, NodeLabelLoader) and \
            not isinstance(label_loader, EdgeLabelLoader):
            raise DGLError("label loader should be a NodeLabelLoader or EdgeLabelLoader.")
        self._label_loader.append(label_loader)

    def addReverseEdge(self):
        pass

    def process(self):
        for feat_loader in self._feature_loader:
            if feat_loader.node_feat:
                node_feat_result = feat_loader.process(self._node_dict)
            else:
                edge_feat_result = feat_loader.process(self._node_dict)

        for edge_loader in self._edge_loader:
            edge_result = edge_loader.process(self._node_dict)

        for label_loader in self._label_loader:
            if label_loader.node_label:
                node, label = label_loader.process(self._node_dict)
            else:
                s_node, d_node, label = label_loader.process(self._node_dict)

    def save(self, path=None):
        pass

    def load(self, path=None):
        pass

    @property
    def node_id_map(self):
        """ Return mappings from internal node id to raw node id/name

        Return
        ------
        dict of dict:
            [node_type →  dict of [graph id(int) → raw node id(string/int)]
        """
        pass

    @property
    def label_map(self):
        """ Return mapping from internal label id to original label

        Return
        ------
        dict:
            dict of [label id(int) → raw label(string/int)]
        """
        pass

    @property
    def graph(self):
        """ Return processed graph
        """
        return self._graph