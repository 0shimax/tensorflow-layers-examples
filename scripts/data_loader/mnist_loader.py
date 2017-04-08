import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def load_data(data_path: str):
    # Load training and eval data
    mnist = input_data.read_data_sets(data_path)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return train_data, train_labels, eval_data, eval_labels
