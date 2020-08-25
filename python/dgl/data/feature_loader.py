"""Classes for loading raw features"""

import os
import csv

from ..base import DGLError, dgl_warning
from .utils import w2v_languages
from .utils import parse_category_single_feat, parse_category_multi_feat
from .utils import parse_numerical_feat
from .utils import parse_numerical_multihot_feat
from .utils import parse_word2vec_feature

def _field2idx(cols, fields):
    idx_cols = []
    # find index of each target field name
    for tg_field in cols:
        for i, field_name in enumerate(fields):
            if field_name == tg_field:
                idx_cols.append(i)
                break
    return idx_cols

class NodeFeatureLoader(object):
    r"""Node feature loader class.

    Load node features from data src.
    The node ids are also inferred when loading the features.

    Parameters
    ----------
    input: str
        Data source, for the csv file input,
        it should be a string of file path
    separator: str, optional
        Delimiter(separator) used in csv file.
        Default: '\t'
    has_header: bool, optional
        Whether the input data has col name.
        Default: True
    int_id: bool, optional
        Whether the raw node id is an int,
        this can help speed things up.
        Default: False
    eager_mode: bool, optional
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
    -----

    * Currently, we only support raw csv file input.

    * If eager_model is True, the loader will processing
    the features immediately after addXXXFeature
    is called. This will case extra performance overhead
    when merging multiple FeatureLoaders together to
    build the DGLGraph. Currently eager_model is not
    supported.

    * If eager_model if False, the features are not
    processed until building the DGLGraph.

    Examples:

    ** Creat a FeatureLoader to load user features from u.csv. **

    >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                             separator="|")
    >>> user_loader.addCategoryFeature(cols=["id", "gender"], node_type='user')
    >>> user_loader.addWord2VecFeature(cols=["id", "occupation"], node_type='user')

    ** Creat a FeatureLoader to load movie features from m.csv. **

    >>> movie_loader = dgl.data.FeatureLoader(input='u.item',
                                              separator="|")
    >>> movie_loader.addMultiHotFeature(cols=["id", "Action", "Adventure",
                                              "Animation", "Children", "Comedy", "Crime",
                                              "Documentary", "Drama", "Fantasy", "Film-Noir",
                                              "Horror", "Musical", "Mystery", "Romance",
                                              "Sci-Fi", "Thriller", "War", "Western"],
                                        norm=True,
                                        node_type='movie')
    >>> movie_loader.addWord2VecFeature(cols=["id", "title"],
                                        language=['en_lang', 'fr_lang'],
                                        node_type='movie')

    ** Append features into graph loader **

    >>> graphloader = dgl.data.GraphLoader()
    >>> graphloader.appendFeature(user_loader)
    >>> graphloader.appendFeature(movie_loader)

    """
    def __init__(self, input, separator='\t', has_head=True, int_id=False, eager_model=False,
        encoding='utf-8', verbose=False):
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
        self._raw_features = []

    def addCategoryFeature(self, cols, rows=None, norm=None, node_type=None):
        r"""Add categorical features for nodes

        Two or more columns of the **input** are chosen, one for
        node name and the others for category data.
        If there is only one column storing the category data,
        it will be encoded using one-hot encoding.
        If there are multiple columns storing the category data,
        they will be encoded using multi-hot encoding.

        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str, ...] column names for node and category.
            The first column is treated as node name and
            the second and the rest columns are treated as category data.
            (2) [int, int, ...] column numbers for node and category.
            The first column is treated as node name and
            the second and the rest columns are treated as category data.

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        norm: str
            Which kind of normalization is applied to the features.
            Supported normalization ops are

            (1) None, do nothing.
            (2) `col`, column-based normalization. Normalize the data
            for each column:

            .. math::
                x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

            (3) `row`, sane as None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Note:

        * Empty string is allowed when there are multiple columns
        storing the category info.

        Example:

        ** Load category features for single category column **

        Example data of u.csv is as follows:
            ====    ======
            name    gender
            ====    ======
            John    M
            Tim     M
            Maggy   F
            ====    ======

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|",
                                                has_head=True)
        >>> user_loader.addCategoryFeature(cols=["name", "gender"])

        ** Load category features for multiple categories **

        Example data of u.csv is as follows:

            ====    ======== ========
            name    role1    role2
            ====    ======== ========
            John    Actor    Director
            Tim     Director Writer
            Maggy   Actor
            ====    ======== ========

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|",
                                                has_head=True)
        >>> user_loader.addCategoryFeature(cols=["name", "role1", "role2"],
                                        norm='col')
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) < 2:
            raise DGLError("addCategoryFeature requires at least 2 columns, " \
                "one for nodes, others for category data")

        nodes = []
        features = []
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

                    if len(cols) == 2:
                        features.append(line[cols[1]])
                    else:
                        row_f = []
                        for i in range(1, len(cols)):
                            # not empty string
                            if line[cols[i]] != "":
                                row_f.append(line[cols[i]])
                        features.append(row_f)
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])

                        if len(cols) == 2:
                            features.append(line[cols[1]])
                        else:
                            row_f = []
                            for i in range(1, len(cols)):
                                # not empty string
                                if line[cols[i]] != "":
                                    row_f.append(line[cols[i]])
                            features.append(row_f)
                            row_idx += 1
                    # else skip this line

        # single category
        if len(cols) == 2:
            feat = parse_category_single_feat(features, norm=norm)
        else:
            feat = parse_category_multi_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((node_type, nodes, feat))

    def addMultiCategoryFeature(self, cols, separator, rows=None, norm=None, node_type=None):
        r"""Add multiple categorical features for nodes

        Two columns of the **input** are chosen, one for
        node name and another for category. The category
        column stores multiple category data in a string,
        e.g., 'Actor,Director'. A separator is required
        to split the category data. The category data
        will be encoded using multi-hot encoding.


        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and category.
            The first column is treated as node name and
            the second column is treated as category data.
            (2) [int, int] column numbers for node and category.
            The first column is treated as node name and
            the second is treated as category data.

        separator: str
            Delimiter(separator) used to split category data.

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        norm: str
            Which kind of normalization is applied to the features.
            Supported normalization ops are

            (1) None, do nothing.
            (2) `col`, column-based normalization. Normalize the data
            for each column:

            .. math::
                x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

            (3) `row`, row-based normalization. Normalize the data for
            each row:

            .. math::
                x_{ij} = \frac{x_{ij}}{\sum_{j=0}^N{x_{ij}}}

            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Note
        -----

        Values in ``rows`` should be in accending order.

        Example
        -------

        ** Load category features **

        Example data of u.csv is as follows:

            ====    ======
            name    role
            ====    ======
            John    Actor,Director
            Tim     Director,Writer
            Maggy   Actor
            ====    ======

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|",
                                                has_head=True)
        >>> user_loader.addMultiCategoryFeature(cols=["name", "role"],
                                                separator=',')
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) == 2:
            raise DGLError("addMultiCategoryFeature only accept two columns, " \
                "one for nodes, another for category data")

        nodes = []
        features = []
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
                    features.append(line[cols[1]].split(separator))
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append(line[cols[1]].split(separator))
                        row_idx += 1
                    # else skip this line

        feat = parse_category_multi_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((node_type, nodes, feat))

    def addNumericalFeature(self, cols, rows=None, norm=None, node_type=None):
        r"""Add numerical features for nodes

        Two columns of the **input** are chosen, one for
        node name and another for numbers. The numbers are
        treated as floats.

        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str, ...] column names for node and numerical data.
            The first column is treated as node name and
            the second and the rest columns are treated as numerical data.
            (2) [int, int, ...] column numbers for node and numerical data.
            The first column is treated as node name and
            the second and the rest columns are treated as numerical data.

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        norm: str
            Which kind of normalization is applied to the features of each column.
            Supported normalization ops are

            (1) None, do nothing.
            (2) `standard`:

            .. math::
                norm_{ij} = \frac{x_{ij}}{\sum_{j=0}^N{|x_{ij}|}}

            (3) `min-max`:

            .. math::
                norm_{ij} = \frac{x_{ij} - min(x_i[:])}{max(x_i[:])-min(x_i[:])}

            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Example:

        ** Load numerical features **

        Example data of u.csv is as follows:

            ====    ======
            name    weight
            ====    ======
            John    120.3
            Tim     100.2
            Maggy   110.5
            ====    ======

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|",
                                                has_head=True)
        >>> user_loader.addNumericalFeature(cols=["name", "weight"])
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) < 2:
            raise DGLError("addNumericalFeature requires at least 2 columns, " \
                "one for nodes, others for numerical data")

        nodes = []
        features = []
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

                    row_f = []
                    for i in range(1, len(cols)):
                        row_f.append(float(line[cols[i]]))
                    features.append(row_f)
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])

                        for i in range(1, len(cols)):
                            row_f.append(float(line[cols[i]]))
                        features.append(row_f)
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((node_type, nodes, feat))

    def addMultiNumericalFeature(self, cols, separator, rows=None, norm=None, node_type=None):
        r"""Add numerical features for nodes

        Two columns of the **input** are chosen, one for
        node name and another for numbers. The numbers are
        treated as floats.

        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and category.
            The first column is treated as node name and
            the second column is treated as category data.
            (2) [int, int] column numbers for node and category.
            The first column is treated as node name and
            the second is treated as category data.

        separator: str
            Delimiter(separator) used to split category data.

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        norm: str
            Which kind of normalization is applied to the features of each column.
            Supported normalization ops are

            (1) None, do nothing.
            (2) `standard`:

            .. math::
                norm_{ij} = \frac{x_{ij}}{\sum_{j=0}^N{|x_{ij}|}}

            (3) `min-max`:

            .. math::
                norm_{ij} = \frac{x_{ij} - min(x_i[:])}{max(x_i[:])-min(x_i[:])}

            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Example:

        ** Load numerical features **

        Example data of u.csv is as follows:

            ====    ======
            name    feature
            ====    ======
            John    1.,2.
            Tim     1.,-1.
            Maggy   2.,3.
            ====    ======

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|",
                                                has_head=True)
        >>> user_loader.addNumericalFeature(cols=["name", "feature"])
        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) == 2:
            raise DGLError("addMultiNumericalFeature only accept two columns, " \
                "one for nodes, another for numerical data")

        nodes = []
        features = []
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
                    features.append([float(val) for val in line[cols[1]].split(separator)])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append([float(val) for val in line[cols[1]].split(separator)])
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((node_type, nodes, feat))

    def addNumericalBucketFeature(self, cols, range, bucket_cnt, slide_window_size=0,
        rows=None, norm=None, node_type=None):
        r""" Add numerical data features for nodes by matching them into
        different buckets.

        Two columns of the **input** are chosen, one for
        node name and another for numbers. A bucket range based
        algorithm is used to convert numerical value into multi-hop
        encoding features.


        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and numerical data.
            The first column is treated as node name and
            the second column is treated as numerical data.
            (2) [int, int] column numbers for node and numerical data.
            The first column is treated as node name and
            the second is treated as numerical data.
        range: list of float or tuple of float
            [low, high]: the range of the numerical value.
            All v_i < low will be set to v_i = low and
            all v_j > high will be set to v_j = high.
        bucket_cnt: int
            Number of bucket to use.
        slide_window_size: int
            The sliding window used to convert numerical value into bucket number.
        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None
        norm: str
            Which kind of normalization is applied to the features.
            Supported normalization ops are

            (1) None, do nothing.
            (2) `col`, column-based normalization. Normalize the data
            for each column.
            (3) `raw`, row-based normalization. Normalize the data for
            each row.
            Default: None
        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Note:

        * The encoding algorithm is as:

        >>> bucket_size = (upper_bound - lower_bound) / bucket_cnt
        >>> low = value - slide_window_size/2
        >>> high = value + slide_window_size/2
        >>> bucket = [i for i in range(math.ceil(low/bucket_size), math.floor(high/bucket_size))]

        Example:

        ** Load numerical data into bucket features **

        Example data of u.csv is as follows:

        ====    ===
        name    age
        ====    ===
        John    21
        Tim     31
        Maggy   55
        ====    ===

        >>> user_loader = dgl.data.FeatureLoader(input='u.csv',
                                                separator="|",
                                                has_head=True)
        >>> user_loader.addNumericalBucketFeature(cols=["name", "age"],
                                                range=[0,100],
                                                bucket_cnt=10,
                                                slide_window_size=5.0)
        ** multi-hop feature of John, Tim and Maggy **

        John: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        Tim:  [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        Maggy:[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) == 2:
            raise DGLError("addNumericalBucketFeature only accept two columns, " \
                "one for nodes, another for numerical data")

        if not isinstance(cols, list) or \
            len(range) != 2 or \
            range[0] >= range[1]:
            raise DGLError("Range is in format of [low, high]. "\
                "low should be smaller than high")

        if bucket_cnt <= 1:
            raise DGLError("Number of bucket should be larger than 1")

        nodes = []
        features = []
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
                    features.append(float(line[cols[1]]))
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append(float(line[cols[1]]))
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_multihot_feat(features,
                                             low=range[0],
                                             high=range[1],
                                             bucket_cnt=bucket_cnt,
                                             window_size=slide_window_size)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((node_type, nodes, feat))

    def addWord2VecFeature(self, cols, languages, rows=None, node_type=None):
        r""" Add word2vec features for nodes

        Two columns of the **input** are chosen, one for
        node name and another for a string.
        By default spacy is used to encode the string.

        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str] column names for node and numerical data.
            The first column is treated as node name and
            the second column is treated as string.
            (2) [int, int] column numbers for node and numerical data.
            The first column is treated as node name and
            the second column is treated as string.

        languages: list of string
            List of languages used to encode the feature string.
            e.g., 'en_core_web_lg', 'fr_core_news_lg'. For more details,
            please see **Notes**. Multiple languages can
            be listed and one embedding vectors are generated for
            each language and are concatenated together.

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        node_type: str
            Node type. If None, default node type is chosen.
            Default: None

        Note:
        -----
        The language model support is backed by scapy

        Example:
        --------

        ** Load words to vector features **

        Example data of u.csv is as follows:

        ====    ===
        name    title
        ====    ===
        Paper1  'Modeling Relational Data with Graph Convolutional     Networks'
        Paper2  'Convolutional neural networks on graphs with fast localized spectral filtering'
        Paper3  'Translating embeddings for modeling multi-relational data'
        ====    ===

        >>> user_loader = dgl.data.FeatureLoader(input='paper.csv',
                                                    separator="|",
                                                    has_head=True)
        >>> user_loader.addNumericalBucketFeature(cols=["name", "title"],
                                                language=['en_lang'],

        """
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) == 2:
            raise DGLError("addWord2VecFeature only accept two columns, " \
                "one for nodes, another for string data")

        nodes = []
        features = []
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
                    features.append(line[cols[1]])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append(line[cols[1]])
                        row_idx += 1
                    # else skip this line

        feat = parse_word2vec_feature(features, languages, self._verbose)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((node_type, nodes, feat))

class EdgeFeatureLoader(object):
    r"""Edge feature loader

    Load edge features from data src.
    The source and destination node ids are also inferred when loading the features.

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

    Note: Currently, we only support raw csv file input.

    If eager_model is True, the loader will processing
    the features immediately after addXXXFeature
    is called. This will case extra performance overhead
    when merging multiple FeatureLoaders together to
    build the DGLGraph.
    If eager_model if False, the features are not
    processed until building the DGLGraph.

    Examples:

    ** Creat a EdgeFeatureLoader to load edge features from data.csv. **

    >>> edge_loader = dgl.data.EdgeFeatureLoader(input='data.csv',
                                                   separator="|")
    >>> edge_loader.addNumericalFeature(cols=["src", "dst", rate"])

        ** Append features into graph loader **
    >>> graphloader = dgl.data.GraphLoader()
    >>> graphloader.appendFeature(edge_loader)

    """
    def __init__(self, input,
        separator='\t', has_head=False, int_id=False, eager_model=False, verbose=False):
        if not os.path.exists(input):
            raise RuntimeError("File not exist {}".format(input))

        assert self._eager_model, "Currently we do not support eager_model"

        self._input = input
        self._separator = separator
        self._has_head = has_head
        self._int_id = int_id
        self._eager_model = eager_mode
        self._verbose = verbose

    def addNumericalFeature(self, cols, rows=None, norm=None, edge_type=None):
        r"""NabeLabelLoader allows users to define the training target or the grand truth of nodes.

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
        if not isinstance(cols, list):
            raise DGLError("The cols should be a list of string or int")

        if len(cols) == 3:
            raise DGLError("addNumericalFeature only accept three columns, " \
                "first two for source and destination nodes, " \
                "the last for numerical data")

        src_nodes = []
        dst_nodes = []
        features = []
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

                    row_f = []
                    for i in range(2, len(cols)):
                        row_f.append(float(line[cols[i]]))
                    features.append(row_f)
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if rows[row_idx] == idx:
                        src_nodes.append(line[cols[0]])
                        dst_nodes.append(line[cols[1]])

                        for i in range(2, len(cols)):
                            row_f.append(float(line[cols[i]]))
                        features.append(row_f)
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_feat(features, norm=None)
        assert len(src_nodes) == feat.shape[0]
        self._raw_features.append((edge_type, src_nodes, dst_nodes, feat))