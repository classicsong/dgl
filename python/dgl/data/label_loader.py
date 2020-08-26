"""Classes for loading node or edge labels"""

import os

from ..base import DGLError, dgl_warning

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

    * If eager_model is True, the loader will processing
    the labels immediately after addXXXSet
    is called. This will case extra performance overhead
    when merging multiple label loaders together to
    build the DGLGraph.

    * If eager_model if False, the labels are not
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
    def __init__(input, separator='\t', has_head=False, int_id=False,
        eager_mode=False, encoding='utf-8', verbose=False):
        if not os.path.exists(input):
            raise RuntimeError("File not exist {}".format(input))

        assert self._eager_model, "Currently we do not support eager_model"

        self._input = input
        self._separator = separator
        self._has_head = has_head
        self._int_id = int_id
        self._eager_model = eager_mode
        self._encoding = encoding
        self._verbose = verbose
        self._label_map = None
        self._labels = []

    def _set_label_map(label_map):
        if self._label_map is None:
            self._label_map = label_map
        else:
            for key, id in label_map:
                assert key in self._label_map

    def _load_labels(cols, multilabel=False, separator=None, rows=None):
        nodes = []
        labels = []
        with open(self._input, newline='', encoding=self._encoding) as csvfile:
            if isinstance(cols[0], str):
                assert self._has_head, \
                    "The column name is provided to identify the target column." \
                    "The input csv should have the head field"
                reader = csv.reader(csvfile, delimiter=self._separator)
                heads = reader.next()
                # find index of each target field name
                idx_cols = _field2idx(cols, heads)

                assert len(idx_cols) == len(cols), \
                    "one or more field names are not found in {}".format(self._input)
                cols = idx_cols
            else:
                reader = csv.reader(csvfile, delimiter=self._separator)
                if self._has_head:
                    # skip field name
                    reader.next()

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

    def addTrainSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
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

    def addValidSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
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

    def addTestSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
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

    def addSet(cols, split_rate, multilabel=False, separator=None, rows=None, node_type=None):
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

    * If eager_model is True, the loader will processing
    the labels immediately after addXXXSet
    is called. This will case extra performance overhead
    when merging multiple label loaders together to
    build the DGLGraph.

    * If eager_model if False, the labels are not
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

    ** Append features into graph loader **
    >>> graphloader = dgl.data.GraphLoader()
    >>> graphloader.appendFeature(user_loader)
    >>> graphloader.appendLabel(label_loader)

    """
    def __init__(input,separator='\t', has_head=False, int_id=False,
        eager_mode=False, verbose=False):
        if not os.path.exists(input):
            raise RuntimeError("File not exist {}".format(input))

        assert self._eager_model, "Currently we do not support eager_model"

        self._input = input
        self._separator = separator
        self._has_head = has_head
        self._int_id = int_id
        self._eager_model = eager_mode
        self._verbose = verbose
        self._label_map = None
        self._labels = []

    def _set_label_map(label_map):
        if self._label_map is None:
            self._label_map = label_map
        else:
            for key, id in label_map:
                assert key in self._label_map

    def _load_labels(cols, multilabel=False, separator=None, rows=None):
        src_nodes = []
        dst_nodes = []
        labels = []
        with open(self._input, newline='', encoding=self._encoding) as csvfile:
            if isinstance(cols[0], str):
                assert self._has_head, \
                    "The column name is provided to identify the target column." \
                    "The input csv should have the head field"
                reader = csv.reader(csvfile, delimiter=self._separator)
                heads = reader.next()
                # find index of each target field name
                idx_cols = _field2idx(cols, heads)

                assert len(idx_cols) == len(cols), \
                    "one or more field names are not found in {}".format(self._input)
                cols = idx_cols
            else:
                reader = csv.reader(csvfile, delimiter=self._separator)
                if self._has_head:
                    # skip field name
                    reader.next()

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

    def addTrainSet(cols, multilabel=False, separator=None, rows=None, edge_type=None):
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

        if multilabel:
            assert len(cols) == 3, "Multi-class label requires one column for labels"
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        src_nodes, dst_nodes, labels, label_map = \
            self._load_labels(cols, multilabel, separator, rows)
        assert len(nodes) == labels.shape[0], \
            'Train nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                       labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (1., 0., 0.)))

    def addValidSet(cols, multilabel=False, separator=None, rows=None, edge_type=None):
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

        if multilabel:
            assert len(cols) == 3, "Multi-class label requires one column for labels"
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        src_nodes, dst_nodes, labels, label_map = \
            self._load_labels(cols, multilabel, separator, rows)
        assert len(nodes) == labels.shape[0], \
            'Valid nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                       labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (0., 1., 0.)))

    def addTestSet(cols, multilabel=False, separator=None, rows=None, edge_type=None):
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

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        src_nodes, dst_nodes, labels, label_map = \
            self._load_labels(cols, multilabel, separator, rows)
        assert len(nodes) == labels.shape[0], \
            'Test nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                      labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (0., 0., 1.)))

    def addSet(cols, split_rate, multilabel=False, separator=None, rows=None, edge_type=None):
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
        assert len(nodes) == labels.shape[0], \
            'nodes shape {} and labels shape {} mismatch'.format(len(nodes),
                                                                 labels.shape[0])
        self._set_label_map(label_map)
        self._labels.append((edge_type,
                             src_nodes,
                             dst_nodes,
                             labels,
                             (split_rate[0], split_rate[1], split_rate[2])))
