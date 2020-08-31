"""Classes for loading node or edge labels"""

import os
import csv

import numpy as np

from ..base import DGLError, dgl_warning
from .utils import parse_category_single_feat, parse_category_multi_feat
from .utils import field2idx, get_id

def split_idx(num_nids, train_split, valid_split, test_split):
    idx = np.arange(num_nids)
    np.random.shuffle(idx)
    train_cnt = int(num_nids * train_split)
    valid_cnt = int(num_nids * valid_split)
    train_idx = idx[:train_cnt]
    valid_idx = idx[train_cnt:train_cnt+valid_cnt]
    test_idx = idx[train_cnt+valid_cnt:]

    return train_idx, valid_idx, test_idx

class NodeLabelLoader(object):
    r"""NabeLabelLoader allows users to define the grand truth of nodes and the
    train/valid/test targets.

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
    encoding: str, optional
        Input file encoding
        Default: 'utf-8'
    verbose: bool, optional
        Whether print debug info during parsing
        Default: False

    Note:

    * Currently, we only support raw csv file input.

    * If eager_mode is True, the loader will processing
    the labels immediately after addXXXSet
    is called. This will case extra performance overhead
    when merging multiple label loaders together to
    build the DGLGraph.

    * If eager_mode if False, the labels are not
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

    ** Append features and labels into graph loader **

    >>> graphloader = dgl.data.GraphLoader()
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
        self.eager_mode = eager_mode
        self._encoding = encoding
        self._verbose = verbose
        self._label_map = None
        self._labels = []

    def _set_label_map(self, label_map):
        if self._label_map is None:
            self._label_map = label_map
        else:
            for key, id in label_map.items():
                assert key in self._label_map

    def _load_labels(self, cols, multilabel=False, separator=None, rows=None):
        nodes = []
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
                    nodes.append(line[cols[0]])
                    if multilabel:
                        labels.append(line[cols[1]].split(separator))
                    else:
                        labels.append(line[cols[1]])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        if multilabel:
                            labels.append(line[cols[1]].split(separator))
                        else:
                            labels.append(line[cols[1]])
                        row_idx += 1
                    # else skip this line
        if multilabel:
            labels, label_map = parse_category_multi_feat(labels, norm=None)
        else:
            labels, label_map = parse_category_single_feat(labels, norm=None)
        return nodes, labels, label_map

    def process(self, node_dicts):
        """ Preparing nodes and labels for creating dgl graph.

        Nodes are converted into consecutive integer ID spaces and
        its corresponding labels are concatenated together.
        """
        results = {}
        for raw_labels in self._labels:
            node_type, nodes, labels, split = raw_labels
            train_split, valid_split, test_split = split
            if node_type in node_dicts:
                nid_map = node_dicts[node_type]
            else:
                nid_map = {}
                node_dicts[node_type] = nid_map

            nids = []
            for node in nodes:
                nid = get_id(nid_map, node)
                nids.append(nid)
            nids = np.asarray(nids)

            # only train
            if train_split == 1.:
                train_nids = nids
                train_labels = labels
                valid_nids, valid_labels = None, None
                test_nids, test_labels = None, None
            # only valid
            elif valid_split == 1.:
                train_nids, train_labels = None, None
                valid_nids = nids
                valid_labels = labels
                test_nids, test_labels = None, None
            # only test
            elif test_split == 1.:
                train_nids, train_labels = None, None
                valid_nids, valid_labels = None, None
                test_nids = nids
                test_labels = labels
            else:
                num_nids = nids.shape[0]
                train_idx, valid_idx, test_idx = \
                    split_idx(num_nids, train_split, valid_split, test_split)
                train_nids = nids[train_idx]
                train_labels = labels[train_idx]
                valid_nids = nids[valid_idx]
                valid_labels = labels[valid_idx]
                test_nids = nids[test_idx]
                test_labels = labels[test_idx]

            # chech if same node_type already exists
            # if so concatenate the labels
            if node_type in results:
                last_train_nids, last_train_labels, \
                last_valid_nids, last_valid_labels, \
                last_test_nids, last_test_labels = results[node_type]

                results[node_type] = (train_nids if last_train_nids is None else \
                                        last_train_nids if train_nids is None else \
                                            np.concatenate((last_train_nids, train_nids)),
                                      train_labels if last_train_labels is None else \
                                        last_train_labels if train_labels is None else \
                                            np.concatenate((last_train_labels, train_labels)),
                                      valid_nids if last_valid_nids is None else \
                                        last_valid_nids if valid_nids is None else \
                                            np.concatenate((last_valid_nids, valid_nids)),
                                      valid_labels if last_valid_labels is None else \
                                        last_valid_labels if valid_labels is None else \
                                            np.concatenate((last_valid_labels, valid_labels)),
                                      test_nids if last_test_nids is None else \
                                        last_test_nids if test_nids is None else \
                                            np.concatenate((last_test_nids, test_nids)),
                                      test_labels if last_test_labels is None else \
                                        last_test_labels if test_labels is None else \
                                            np.concatenate((last_test_labels, test_labels)))
            else:
                results[node_type] = (train_nids, train_labels,
                                      valid_nids, valid_labels,
                                      test_nids, test_labels)
        return results

    def addTrainSet(self, cols, multilabel=False, separator=None, rows=None, node_type=None):
        r"""Add Training Set.

        Two columns of the **input** are chosen, one for
        node name and another for label string. Multi-label
        is supported, but a separator is required to split
        the labels.

        cols: list of str or list of int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and labels.
            The first column is treated as node name and
            the second column is treated as label.
            (2) [int, int] column numbers for node and labels.
            The first column is treated as node name and
            the second is treated as label.

        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Examples:

        ** Load train labels **

        Example data of label.csv is as follows:

        ====    ======
        name    label
        ====    ======
        John    Actor,Director
        Tim     Director,Writer
        Maggy   Actor
        ====    ======

        >>> label_loader = dgl.data.NodeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addTrainSet(['name', 'label'],
                                    multilabel=True,
                                    separator=','
                                    rows=np.arange(start=0, stop=100))
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addTrainSet only accept two columns, one for nodes, another for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        nodes, labels, label_map = self._load_labels(cols, multilabel, separator, rows)
        assert len(nodes) == labels.shape[0], \
            'Train nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                       labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((node_type, nodes, labels, (1., 0., 0.)))

    def addValidSet(self, cols, multilabel=False, separator=None, rows=None, node_type=None):
        r"""Add Validation Set.

        Two columns of the **input** are chosen, one for
        node name and another for label string. Multi-label
        is supported, but a separator is required to split
        the labels.

        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and labels.
            The first column is treated as node name and
            the second column is treated as label.
            (2) [int, int] column numbers for node and labels.
            The first column is treated as node name and
            the second is treated as label.

        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Examples:

        ** Load valid labels **

        Example data of label.csv is as follows:

        ====    ======
        name    label
        ====    ======
        John    Actor,Director
        Tim     Director,Writer
        Maggy   Actor
        ====    ======

        >>> label_loader = dgl.data.NodeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addValidSet(['name', 'label'],
                                    multilabel=True,
                                    separator=','
                                    rows=np.arange(start=100, stop=120))
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addValidSet only accept two columns, one for nodes, another for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        nodes, labels, label_map = self._load_labels(cols, multilabel, separator, rows)
        assert len(nodes) == labels.shape[0], \
            'Valid nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                       labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((node_type, nodes, labels, (0., 1., 0.)))

    def addTestSet(self, cols, multilabel=False, separator=None, rows=None, node_type=None):
        r"""Add Test Set.

        Two columns of the **input** are chosen, one for
        node name and another for label string. Multi-label
        is supported, but a separator is required to split
        the labels.

        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and labels.
            The first column is treated as node name and
            the second column is treated as label.
            (2) [int, int] column numbers for node and labels.
            The first column is treated as node name and
            the second is treated as label.

        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Examples:

        ** Load test labels **

        Example data of label.csv is as follows:

        ====    ======
        name    label
        ====    ======
        John    Actor,Director
        Tim     Director,Writer
        Maggy   Actor
        ====    ======

        >>> label_loader = dgl.data.NodeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addTestSet(['name', 'label'],
                                    multilabel=True,
                                    separator=','
                                    rows=np.arange(start=120, stop=130))
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addTestSet only accept two columns, one for nodes, another for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        nodes, labels, label_map = self._load_labels(cols, multilabel, separator, rows)
        assert len(nodes) == labels.shape[0], \
            'Test nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                      labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((node_type, nodes, labels, (0., 0., 1.)))

    def addSet(self, cols, split_rate, multilabel=False, separator=None, rows=None, node_type=None):
        r"""Add Test Set.

        Two columns of the **input** are chosen, one for
        node name and another for label string. Multi-label
        is supported, but a separator is required to split
        the labels.

        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and labels.
            The first column is treated as node name and
            the second column is treated as label.
            (2) [int, int] column numbers for node and labels.
            The first column is treated as node name and
            the second is treated as label.

        split_rate: triple of float
            [train, valid, test]: Random split rate, train + valid + test = 1.0, any of train, valid and test can be 0.0


        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Examples:

        ** Load labels **

        Example data of label.csv is as follows:

        ====    ======
        name    label
        ====    ======
        John    Actor,Director
        Tim     Director,Writer
        Maggy   Actor
        ====    ======

        >>> label_loader = dgl.data.NodeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addSet(['name', 'label'],
                                split_rate=[0.7,0.2,0.1],
                                multilabel=True,
                                separator=',')
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addSet only accept two columns, one for nodes, another for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        if not isinstance(split_rate, list) or len(split_rate) != 3:
            raise DGLError("The split_rate should be a list of three floats")
        if split_rate[0] < 0 or split_rate[1] < 0 or split_rate[2] < 0:
            raise DGLError("Split rates must >= 0.")
        if split_rate[0] + split_rate[1] + split_rate[2] != 1.:
            raise DGLError("The sum of split rates should be 1.")

        nodes, labels, label_map = self._load_labels(cols, multilabel, separator, rows)
        assert len(nodes) == labels.shape[0], \
            'nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                 labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((node_type,
                             nodes,
                             labels,
                             (split_rate[0], split_rate[1], split_rate[2])))

    @property
    def node_label(self):
        """ This is node label loader
        """
        return True

    @property
    def label_map(self):
        """ Get the label map
        """
        return self._label_map

class EdgeLabelLoader(object):
    r"""EdgeLabelLoader allows users to define the grand truth of nodes and the
    train/valid/test targets.

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
    the labels immediately after addXXXSet
    is called. This will case extra performance overhead
    when merging multiple label loaders together to
    build the DGLGraph.

    * If eager_mode if False, the labels are not
    processed until building the DGLGraph.

    Examples:

    ** Creat a FeatureLoader to load user features from u.csv.**

    >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                             separator="|")
    >>> user_loader.addCategoryFeature(cols=["id", "gender"], node_type='user')

    ** create node label loader to load labels **

    >>> label_loader = dgl.data.EdgeLabelLoader(input='label.csv',
                                                separator="|")
    >>> label_loader.addTrainSet([0, 1, 2], rows=np.arange(start=0,
                                                           stop=100))

    ** Append features into graph loader **
    >>> graphloader = dgl.data.GraphLoader()
    >>> graphloader.appendFeature(user_loader)
    >>> graphloader.appendLabel(label_loader)

    """
    def __init__(self, input,separator='\t', has_head=True, int_id=False,
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
        self._label_map = None
        self._labels = []

    def _set_label_map(self, label_map):
        if self._label_map is None:
            self._label_map = label_map
        else:
            for key, id in label_map.items():
                assert key in self._label_map

    def _load_labels(self, cols, multilabel=False, separator=None, rows=None):
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
                    if multilabel:
                        labels.append(line[cols[2]].split(separator))
                    else:
                        if len(cols) == 3:
                            labels.append(line[cols[2]])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        src_nodes.append(line[cols[0]])
                        dst_nodes.append(line[cols[1]])
                        if multilabel:
                            labels.append(line[cols[2]].split(separator))
                        else:
                            if len(cols) == 3:
                                labels.append(line[cols[2]])
                        row_idx += 1
                    # else skip this line
        if multilabel:
            labels, label_map = parse_category_multi_feat(labels, norm=None)
        else:
            if len(cols) == 3:
                labels, label_map = parse_category_single_feat(labels, norm=None)
            else:
                labels = None
                label_map = None
        return src_nodes, dst_nodes, labels, label_map

    def process(self, node_dicts):
        """ Preparing edges and labels for creating dgl graph.

        Src nodes and dst nodes are converted into consecutive integer ID spaces and
        its corresponding labels are concatenated together.
        """
        results = {}
        for raw_labels in self._labels:
            edge_type, src_nodes, dst_nodes, labels, split = raw_labels
            train_split, valid_split, test_split = split
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

            # only train
            if train_split == 1.:
                train_snids = snids
                train_dnids = dnids
                train_labels = labels
                valid_snids, valid_dnids, valid_labels = None, None, None
                test_snids, test_dnids, test_labels = None, None, None
            # only valid
            elif valid_split == 1.:
                train_snids, train_dnids, train_labels = None, None, None
                valid_snids = snids
                valid_dnids = dnids
                valid_labels = labels
                test_snids, test_dnids, test_labels = None, None, None
            # only test
            elif test_split == 1.:
                train_snids, train_dnids, train_labels = None, None, None
                valid_snids, valid_dnids, valid_labels = None, None, None
                test_snids = snids
                test_dnids = dnids
                test_labels = labels
            else:
                num_nids = snids.shape[0]
                train_idx, valid_idx, test_idx = \
                    split_idx(num_nids, train_split, valid_split, test_split)
                train_snids = snids[train_idx]
                train_dnids = dnids[train_idx]
                valid_snids = snids[valid_idx]
                valid_dnids = dnids[valid_idx]
                test_snids = snids[test_idx]
                test_dnids = dnids[test_idx]
                if len(labels) > 0:
                    train_labels = labels[train_idx]
                    valid_labels = labels[valid_idx]
                    test_labels = labels[test_idx]
                else:
                    train_labels, valid_labels, test_labels = None, None, None

            # chech if same edge_type already exists
            # if so concatenate the labels
            if edge_type in results:
                last_train_snids, last_train_dnids, last_train_labels, \
                last_valid_snids, last_valid_dnids, last_valid_labels, \
                last_test_snids, last_test_dnids, last_test_labels = results[edge_type]

                results[edge_type] = (train_snids if last_train_snids is None else \
                                        last_train_snids if train_snids is None else \
                                            np.concatenate((last_train_snids, train_snids)),
                                      train_dnids if last_train_dnids is None else \
                                        last_train_dnids if train_dnids is None else \
                                            np.concatenate((last_train_dnids, train_dnids)),
                                      train_labels if last_train_labels is None else \
                                        last_train_labels if train_labels is None else \
                                            np.concatenate((last_train_labels, train_labels)),
                                      valid_snids if last_valid_snids is None else \
                                        last_valid_snids if valid_snids is None else \
                                            np.concatenate((last_valid_snids, valid_snids)),
                                      valid_dnids if last_valid_dnids is None else \
                                        last_valid_dnids if valid_dnids is None else \
                                            np.concatenate((last_valid_dnids, valid_dnids)),
                                      valid_labels if last_valid_labels is None else \
                                        last_valid_labels if valid_labels is None else \
                                        np.concatenate((last_valid_labels, valid_labels)),
                                      test_snids if last_test_snids is None else \
                                        last_test_snids if test_snids is None else \
                                            np.concatenate((last_test_snids, test_snids)),
                                      test_dnids if last_test_dnids is None else \
                                        last_test_dnids if test_dnids is None else \
                                            np.concatenate((last_test_dnids, test_dnids)),
                                      test_labels if last_test_labels is None else \
                                        last_test_labels if test_labels is None else \
                                             np.concatenate((last_test_labels, test_labels)))
            else:
                results[edge_type] = (train_snids, train_dnids, train_labels,
                                      valid_snids, valid_dnids, valid_labels,
                                      test_snids, test_dnids, test_labels)
        return results

    def addTrainSet(self, cols, multilabel=False, separator=None, rows=None, edge_type=None):
        r"""Add Training Set.

        Two or three columns of the **input** are chosen.

        If only two columns are provied, they represent the
        column names of the source nodes and destination nodes.
        This represents the existance of the edges.

        If three columns are provided, the first two columns
        represent the column names of the source nodes and
        destination nodes while the last column give the labels.
        Multi-label is supported, but a separator is required to
        split the labels.

        cols: list of str or list of int
            Which columns to use. Supported data formats are:

            (1) [str, str] column names for source node, destination node.
            (2) [int, int] column numbers for source node, destination node.
            (3) [str, str, str] column names for source node, destination node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.
            (4) [int, int, int] column numbers for node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.

        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        edge_type: str
            Canonical edge type. If None, default edge type is chosen.
            Default: None

        Examples:

        ** Load train labels **

        Example data of label.csv is as follows:

        ====    ========  ====
        name    movie     rate
        ====    ========  ====
        John    StarWar1  5.0
        Tim     X-Man     3.5
        Maggy   StarWar1  4.5
        ====    ========  ====

        >>> label_loader = dgl.data.EdgeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addTrainSet(['name', 'movie', 'rate'],
                                    rows=np.arange(start=0, stop=100))
        """

        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            raise DGLError("addTrainSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if edge_type != None and len(edge_type) != 3:
            raise DGLError("edge_type should be None or a tuple of " \
                "(src_type, relation_type, dst_type)")

        if multilabel:
            assert len(cols) == 3, "Multi-class label requires one column for labels"
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        src_nodes, dst_nodes, labels, label_map = \
            self._load_labels(cols, multilabel, separator, rows)
        if len(cols) == 3:
            assert len(src_nodes) == labels.shape[0], \
                'Train nodes shape {} and labels shape {} mismatch'.format(len(src_nodes),
                                                                        labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (1., 0., 0.)))

    def addValidSet(self, cols, multilabel=False, separator=None, rows=None, edge_type=None):
        r"""Add Validation Set.

        Two or three columns of the **input** are chosen.

        If only two columns are provied, they represent the
        column names of the source nodes and destination nodes.
        This represents the existance of the edges.

        If three columns are provided, the first two columns
        represent the column names of the source nodes and
        destination nodes while the last column give the labels.
        Multi-label is supported, but a separator is required to
        split the labels.

        cols: list of str or list of int
            Which columns to use. Supported data formats are:

            (1) [str, str] column names for source node, destination node.
            (2) [int, int] column numbers for source node, destination node.
            (3) [str, str, str] column names for source node, destination node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.
            (4) [int, int, int] column numbers for node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.

        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        edge_type: str
            Canonical edge type. If None, default edge type is chosen.
            Default: None

        Examples:

        ** Load valid labels **

        Example data of label.csv is as follows:

        ====    ========  ====
        name    movie     rate
        ====    ========  ====
        John    StarWar1  5.0
        Tim     X-Man     3.5
        Maggy   StarWar1  4.5
        ====    ========  ====

        >>> label_loader = dgl.data.EdgeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addValidSet(['name', 'movie', 'rate'],
                                    rows=np.arange(start=0, stop=100))

        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            raise DGLError("addValidSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if edge_type != None and len(edge_type) != 3:
            raise DGLError("edge_type should be None or a tuple of " \
                "(src_type, relation_type, dst_type)")

        if multilabel:
            assert len(cols) == 3, "Multi-class label requires one column for labels"
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        src_nodes, dst_nodes, labels, label_map = \
            self._load_labels(cols, multilabel, separator, rows)
        if len(cols) == 3:
            assert len(src_nodes) == labels.shape[0], \
                'Valid nodes shape {} and labels shape {} mismatch'.format(len(src_nodes),
                                                                        labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (0., 1., 0.)))

    def addTestSet(self, cols, multilabel=False, separator=None, rows=None, edge_type=None):
        r"""Add Test Set.

        Two or three columns of the **input** are chosen.

        If only two columns are provied, they represent the
        column names of the source nodes and destination nodes.
        This represents the existance of the edges.

        If three columns are provided, the first two columns
        represent the column names of the source nodes and
        destination nodes while the last column give the labels.
        Multi-label is supported, but a separator is required to
        split the labels.

        cols: list of str or list of int
            Which columns to use. Supported data formats are:

            (1) [str, str] column names for source node, destination node.
            (2) [int, int] column numbers for source node, destination node.
            (3) [str, str, str] column names for source node, destination node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.
            (4) [int, int, int] column numbers for node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.

        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        edge_type: str
            Canonical edge type. If None, default edge type is chosen.
            Default: None

        Examples:

        ** Load test labels **

        Example data of label.csv is as follows:

        ====    ========  ====
        name    movie     rate
        ====    ========  ====
        John    StarWar1  5.0
        Tim     X-Man     3.5
        Maggy   StarWar1  4.5
        ====    ========  ====

        >>> label_loader = dgl.data.EdgeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addTestSet(['name', 'movie', 'rate'],
                                    rows=np.arange(start=0, stop=100))
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            assert len(cols) == 3, "Multi-class label requires one column for labels"
            raise DGLError("addTestSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if edge_type != None and len(edge_type) != 3:
            raise DGLError("edge_type should be None or a tuple of " \
                "(src_type, relation_type, dst_type)")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        src_nodes, dst_nodes, labels, label_map = \
            self._load_labels(cols, multilabel, separator, rows)
        if len(cols) == 3:
            assert len(src_nodes) == labels.shape[0], \
                'Test nodes shape {} and labels shape {} mismatch'.format(len(src_nodes),
                                                                        labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (0., 0., 1.)))

    def addSet(self, cols, split_rate, multilabel=False, separator=None, rows=None, edge_type=None):
        r"""Add Test Set.

        Two or three columns of the **input** are chosen.

        If only two columns are provied, they represent the
        column names of the source nodes and destination nodes.
        This represents the existance of the edges.

        If three columns are provided, the first two columns
        represent the column names of the source nodes and
        destination nodes while the last column give the labels.
        Multi-label is supported, but a separator is required to
        split the labels.

        cols: list of str or list of int
            Which columns to use. Supported data formats are:

            (1) [str, str] column names for source node, destination node.
            (2) [int, int] column numbers for source node, destination node.
            (3) [str, str, str] column names for source node, destination node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.
            (4) [int, int, int] column numbers for node and labels.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as label.

        split_rate: triple of float
            [train, valid, test]: Random split rate, train + valid + test = 1.0, any of train, valid and test can be 0.0


        multilabel: bool
            Whether it is a multi-label task.
            Default: False

        separator: str, optional
            Delimiter(separator) used to split label data.
            Default: None

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        edge_type: str
            Canonical edge type. If None, default edge type is chosen.
            Default: None

        Examples:

        ** Load labels **

        Example data of label.csv is as follows:

        ====    ========  ====
        name    movie     rate
        ====    ========  ====
        John    StarWar1  5.0
        Tim     X-Man     3.5
        Maggy   StarWar1  4.5
        ====    ========  ====

        >>> label_loader = dgl.data.EdgeLabelLoader(input='label.csv',
                                                    separator="\t")
        >>> label_loader.addSet(['name', 'movie', 'rate'],
                                rows=np.arange(start=0, stop=100),
                                split_rate=[0.7,0.2,0.1])
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            raise DGLError("addSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if edge_type != None and len(edge_type) != 3:
            raise DGLError("edge_type should be None or a tuple of " \
                "(src_type, relation_type, dst_type)")

        if multilabel:
            assert len(cols) == 3, "Multi-class label requires one column for labels"
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        if not isinstance(split_rate, list) or len(split_rate) != 3:
            raise DGLError("The split_rate should be a list of three floats")
        if split_rate[0] < 0 or split_rate[1] < 0 or split_rate[2] < 0:
            raise DGLError("Split rates must >= 0.")
        if split_rate[0] + split_rate[1] + split_rate[2] != 1.:
            raise DGLError("The sum of split rates should be 1.")

        src_nodes, dst_nodes, labels, label_map = \
            self._load_labels(cols, multilabel, separator, rows)
        if len(cols) == 3:
            assert len(src_nodes) == labels.shape[0], \
                'nodes shape {} and labels shape {} mismatch'.format(len(src_nodes),
                                                                    labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (split_rate[0], split_rate[1], split_rate[2])))

    @property
    def node_label(self):
        """ This is edge label loader
        """
        return False

    @property
    def label_map(self):
        """ Get the label map
        """
        return self._label_map
