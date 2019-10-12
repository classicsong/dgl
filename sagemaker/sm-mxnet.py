# S3 prefix
prefix = 'kg-example-fb15k'

import boto3
import re

import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role

import sagemaker as sage
from time import gmtime, strftime

# Setup session
sess = sage.Session()

# Get role
role = '<your-sagemaker-role>'

# setup data
WORK_DIRECTORY = './ml/input/data/'
data_location = sess.upload_data(WORK_DIRECTORY, key_prefix=prefix)

image = '<your-ecr>/dgl-gpu-dlc:kg-fb15k-distmult-mxnet-example'

tree = sage.estimator.Estimator(image,
                       role, 1, 'ml.p3.2xlarge',
                       output_path="s3://<your-s3>/output".format(sess.default_bucket()),
                       sagemaker_session=sess)

tree.fit(data_location)


