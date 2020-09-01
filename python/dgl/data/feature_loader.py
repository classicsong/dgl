"""Classes for loading raw features"""

import os
import csv
import numpy as np

from .utils import parse_category_single_feat, parse_category_multi_feat
from .utils import parse_numerical_feat
from .utils import parse_numerical_multihot_feat
from .utils import parse_word2vec_feature
from .utils import field2idx, get_id

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

    * Here we assume features created from different columns
    are in the same order, thus we can concatenate them directly.

    * If eager_mode is True, the loader will processing
    the features immediately after addXXXFeature
    is called. This will case extra performance overhead
    when merging multiple FeatureLoaders together to
    build the DGLGraph. Currently eager_mode is not
    supported.

    * If eager_mode if False, the features are not
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
    def __init__(self, input, separator='\t', has_head=True, int_id=False, eager_mode=False,
        encoding='utf-8', verbose=False):
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
        self._raw_features = []

    def process(self, node_dicts):
        """ Preparing nodes and features for creating dgl graph.

        Nodes are converted into consecutive integer ID spaces and
        its corresponding features are concatenated together.

        Params:
        node_dicts: dict of dict
            {node_type: {node_str : node_id}}

        Return:
            dict
            {node_type: (node_ids, node_feats)}
        """
        results = {}
        for raw_feat in self._raw_features:
            feat_name, node_type, nodes, feats = raw_feat
            if node_type in node_dicts:
                nid_map = node_dicts[node_type]
            else:
                nid_map = {}
                node_dicts[node_type] = nid_map

            nids = []
            for node in nodes:
                nid = get_id(nid_map, node)
                nids.append(nid)
            nids = np.asarray(nids, dtype='long')

            # chech if same node_type already exists
            # if so concatenate the features.
            if node_type in results:
                if feat_name in results[node_type]:
                    last_nids, last_feats = results[node_type][feat_name]
                    assert last_nids.shape[0] == nids.shape[0], \
                        "Input features from different columns should have the same shape." \
                        "but got {} vs. {}".format(last_nids.shape[0], nids.shape[0])
                    results[node_type][feat_name] = (last_nids,
                                                     np.concatenate((last_feats, feats), axis=1))
                else:
                    results[node_type][feat_name] = (nids, feats)
            else:
                results[node_type] = {}
                results[node_type][feat_name] = (nids, feats)

        return results

    def addCategoryFeature(self, cols, feat_name='nf', rows=None, norm=None, node_type=None):
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

        feat_name: str
            Feature name.
            Default: nf

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
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) < 2:
            raise RuntimeError("addCategoryFeature requires at least 2 columns, " \
                "one for nodes, others for category data")

        if self._verbose:
            print('NodeFeatureLoader.addCategoryFeature " \
                  "from {} with {}'.format(self._input, cols))

        nodes = []
        features = []
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
                    if len(rows) == row_idx:
                        break
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
            feat, _ = parse_category_single_feat(features, norm=norm)
        else:
            feat, _ = parse_category_multi_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((feat_name, node_type, nodes, feat))

    def addMultiCategoryFeature(self, cols, separator, feat_name='nf', rows=None, norm=None, node_type=None):
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

        feat_name: str
            Feature name.
            Default: nf

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
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise RuntimeError("addMultiCategoryFeature only accept two columns, " \
                "one for nodes, another for category data")

        if self._verbose:
            print('NodeFeatureLoader.addCategoryFeature " \
                  "from {} with {}'.format(self._input, cols))

        nodes = []
        features = []
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
                    features.append(line[cols[1]].split(separator))
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append(line[cols[1]].split(separator))
                        row_idx += 1
                    # else skip this line

        feat, _ = parse_category_multi_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((feat_name, node_type, nodes, feat))

    def addNumericalFeature(self, cols, feat_name='nf', rows=None, norm=None, node_type=None):
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

        feat_name: str
            Feature name.
            Default: nf

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
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) < 2:
            raise RuntimeError("addNumericalFeature requires at least 2 columns, " \
                "one for nodes, others for numerical data")

        if self._verbose:
            print('NodeFeatureLoader.addCategoryFeature " \
                  "from {} with {}'.format(self._input, cols))

        nodes = []
        features = []
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

                    row_f = []
                    for i in range(1, len(cols)):
                        row_f.append(float(line[cols[i]]))
                    features.append(row_f)
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])

                        row_f = []
                        for i in range(1, len(cols)):
                            row_f.append(float(line[cols[i]]))
                        features.append(row_f)
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((feat_name, node_type, nodes, feat))

    def addMultiNumericalFeature(self, cols, feat_name='nf', separator, rows=None, norm=None, node_type=None):
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

        feat_name: str
            Feature name.
            Default: nf

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
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise RuntimeError("addMultiNumericalFeature only accept two columns, " \
                "one for nodes, another for numerical data")

        if self._verbose:
            print('NodeFeatureLoader.addCategoryFeature " \
                  "from {} with {}'.format(self._input, cols))

        nodes = []
        features = []
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
                    features.append([float(val) for val in line[cols[1]].split(separator)])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append([float(val) for val in line[cols[1]].split(separator)])
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_feat(features, norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((feat_name, node_type, nodes, feat))

    def addNumericalBucketFeature(self, cols, range, bucket_cnt, feat_name='nf', slide_window_size=0,
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

        feat_name: str
            Feature name.
            Default: nf

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
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise RuntimeError("addNumericalBucketFeature only accept two columns, " \
                "one for nodes, another for numerical data")

        if not isinstance(cols, list) or \
            len(range) != 2 or \
            range[0] >= range[1]:
            raise RuntimeError("Range is in format of [low, high]. "\
                "low should be smaller than high")

        if bucket_cnt <= 1:
            raise RuntimeError("Number of bucket should be larger than 1")

        if self._verbose:
            print('NodeFeatureLoader.addCategoryFeature " \
                  "from {} with {}'.format(self._input, cols))

        nodes = []
        features = []
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
                    features.append(float(line[cols[1]]))
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append(float(line[cols[1]]))
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_multihot_feat(features,
                                             low=range[0],
                                             high=range[1],
                                             bucket_cnt=bucket_cnt,
                                             window_size=slide_window_size,
                                             norm=norm)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((feat_name, node_type, nodes, feat))

    def addWord2VecFeature(self, cols, languages, feat_name='nf', rows=None, node_type=None):
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

        feat_name: str
            Feature name.
            Default: nf

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
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) != 2:
            raise RuntimeError("addWord2VecFeature only accept two columns, " \
                "one for nodes, another for string data")

        if self._verbose:
            print('NodeFeatureLoader.addCategoryFeature " \
                  "from {} with {}'.format(self._input, cols))

        nodes = []
        features = []
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
                    features.append(line[cols[1]])
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        nodes.append(line[cols[0]])
                        features.append(line[cols[1]])
                        row_idx += 1
                    # else skip this line

        feat = parse_word2vec_feature(features, languages, self._verbose)
        assert len(nodes) == feat.shape[0]
        self._raw_features.append((feat_name, node_type, nodes, feat))

    @property
    def node_feat(self):
        """ This is node feature loader
        """
        return True

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

    If eager_mode is True, the loader will processing
    the features immediately after addXXXFeature
    is called. This will case extra performance overhead
    when merging multiple FeatureLoaders together to
    build the DGLGraph.
    If eager_mode if False, the features are not
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
    def __init__(self, input, separator='\t', has_head=True, int_id=False, eager_mode=False,
        encoding='utf-8', verbose=False):
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
        self._raw_features = []

    def process(self, node_dicts):
        """ Preparing edges and features for creating dgl graph.

        Src and dst nodes are converted into consecutive integer ID spaces and
        the edge features are concatenated together.

        Params:
        node_dicts: dict of dict
            {node_type: {node_str : node_id}}

        Return:
            dict
            {edge_type: (snids, dnids, edge_feats)}
        """
        results = {}
        for raw_feat in self._raw_features:
            feat_name, edge_type, src_nodes, dst_nodes, feats = raw_feat
            if edge_type is None:
                src_type = None
                dst_type = None
            else:
                src_type, rel_type, dst_type = edge_type
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

            # chech if same edge_type already exists
            # if so concatenate the features.
            if edge_type in results:
                if feat_name in results[node_type]:
                    last_snids, last_dnids, last_feats = results[edge_type][feat_name]
                    assert last_snids.shape[0] == snids.shape[0], \
                        "Input features from different columns should have the same shape." \
                        "but got {} vs. {}".format(last_snids.shape[0], snids.shape[0])
                    results[edge_type][feat_name] = (last_snids, last_dnids,
                                                    np.concatenate((last_feats, feats), axis=1))
                else:
                    results[edge_type][feat_name] = (snids, dnids, feats)
            else:
                results[edge_type] = {}
                results[edge_type][feat_name] = (snids, dnids, feats)
        return results

    def addNumericalFeature(self, cols, feat_name='ef', rows=None, norm=None, edge_type=None):
        r"""Add numerical features for nodes

        Three columns of the **input** are chosen, one for
        source node name, one for destination node name
        and the last for numbers. The numbers are
        treated as floats.

        Parameters
        ----------
        cols: list of str or int
            Which columns to use. Supported data formats are

            (1) [str, str, str] column names for node and numerical data.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as numerical data.
            (2) [int, int, int] column numbers for node and numerical data.
            The first column is treated as source node name,
            the second column is treated as destination node name and
            the third column is treated as numerical data.

        feat_name: str
            Feature name.
            Default: ef

        rows: numpy.array or list of int
            Which row(s) to load. None to load all.
            Default: None

        norm: str
            Which kind of normalization is applied to the features.
            Supported normalization ops are

            (1) None, do nothing.
            (2) `standard`:

            .. math::
                norm_i = \frac{x_i}{\sum_{i=0}^N{x_i}}

            (3) `min-max`:

            .. math::
                norm_i = \frac{x_i - min(x[:])}{max(x[:])-min(x[:])}

            Default: None

        edge_type: triple of str
            Canonical edge type, should be in format of (src_type, rel_type, dst_type).
            If None, default canonical edge type is chosen.
            Default: None

        Example:

        ** Load numerical features **

        Example data of data.csv is as follows:

        ====    ========  ====
        name    movie     rate
        ====    ========  ====
        John    StarWar1  5.0
        Tim     X-Man     3.5
        Maggy   StarWar1  4.5
        ====    ========  ====

        >>> user_loader = dgl.data.FeatureLoader(input='data.csv',
                                                    separator="|",
                                                    has_head=True)
        >>> user_loader.addNumericalFeature(cols=["name", "movie", "rate"])
        """
        if not isinstance(cols, list):
            raise RuntimeError("The cols should be a list of string or int")

        if len(cols) != 3:
            raise RuntimeError("addNumericalFeature only accept three columns, " \
                "first two for source and destination nodes, " \
                "the last for numerical data")

        if self._verbose:
            print('EdgeFeatureLoader.addNumericalFeature " \
                  "from {} with {}'.format(self._input, cols))

        src_nodes = []
        dst_nodes = []
        features = []
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

                    row_f = []
                    for i in range(2, len(cols)):
                        row_f.append(float(line[cols[i]]))
                    features.append(row_f)
            else:
                row_idx = 0
                for idx, line in enumerate(reader):
                    if len(rows) == row_idx:
                        break
                    if rows[row_idx] == idx:
                        src_nodes.append(line[cols[0]])
                        dst_nodes.append(line[cols[1]])

                        row_f = []
                        for i in range(2, len(cols)):
                            row_f.append(float(line[cols[i]]))
                        features.append(row_f)
                        row_idx += 1
                    # else skip this line

        feat = parse_numerical_feat(features, norm=norm)
        assert len(src_nodes) == feat.shape[0]
        self._raw_features.append((feat_name, edge_type, src_nodes, dst_nodes, feat))

    @property
    def node_feat(self):
        """ This is edge feature loader
        """
        return False
