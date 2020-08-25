""" Classes for loading raw graph"""

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

    * If eager_model is True, the loader will processing
    the edges immediately after addXXXEdges
    is called. This will case extra performance overhead
    when merging multiple edge loaders together to
    build the DGLGraph.

    * If eager_model if False, the edges are not
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

    def addEdges(self, cols, rows=None, src_type=None, dst_type=None, edge_type=None):
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise DGLError("addEdges only accepts two columns "\
                           "for source node and destination node.")

        # TODO(xiang)

    def addCategoryRelation(self, cols, rows=None, src_type=None, dst_type=None):
        r"""Convert categorical features into edge relations

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

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        src_type: str
            Source node type. If None, default source node type is chosen.
            Default: None

        dst_type: str
            Destination node type. If None, default destination node type is chosen.
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
        >>> edgeloader.addCategoryRelation(cols=["name", "rate", "movie"], src_type='user', dst_type='movie')

        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) != 3:
            raise DGLError("addCategoryRelation only accepts three columns " \
                           "the first column for source node, " \
                           "the second for destination node, " \
                           "and third for edge category")

        # TODO(xiang)

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
        pass

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