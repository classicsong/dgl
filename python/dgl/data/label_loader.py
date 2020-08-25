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
    def __init__(input, separator='\t', has_head=False, int_id=False,
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

    def addTrainSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addTrainSet only accept two columns, one for nodes, another for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        # TODO(Xiang):


    def addValidSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addValidSet only accept two columns, one for nodes, another for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        # TODO(Xiang)

    def addTestSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addTestSet only accept two columns, one for nodes, another for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        # TODO(Xiang)

    def addSet(cols, split_rate, multilabel=False, separator=None, rows=None, node_type=None):
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

        # TODO(Xiang)

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

    def addTrainSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            raise DGLError("addTrainSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        # TODO(Xiang):


    def addValidSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            raise DGLError("addValidSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        # TODO(Xiang)

    def addTestSet(cols, multilabel=False, separator=None, rows=None, node_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            raise DGLError("addTestSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        # TODO(Xiang)

    def addSet(cols, split_rate, multilabel=False, separator=None, rows=None, node_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2 and len(cols) != 3:
            raise DGLError("addSet accepts two columns " \
                           "for source node and destination node." \
                           "or three columns, the first column for source node, " \
                           "the second for destination node, " \
                           "and third for labels")

        if multilabel:
            assert separator is not None, "Multi-class label is supported, "\
                "but a separator is required to split the labels"

        if not isinstance(split_rate, list) or len(split_rate) != 3:
            raise DGLError("The split_rate should be a list of three floats")
        if split_rate[0] < 0 or split_rate[1] < 0 or split_rate[2] < 0:
            raise DGLError("Split rates must >= 0.")
        if split_rate[0] + split_rate[1] + split_rate[2] != 1.:
            raise DGLError("The sum of split rates should be 1.")

        # TODO(Xiang)
