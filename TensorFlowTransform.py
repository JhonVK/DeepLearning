from __future__ import print_function
import tempfile
import pandas as pd

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam

import apache_beam.io.iobase

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

dataset = pd.read_csv("pollution-small.csv")
print(dataset.head())