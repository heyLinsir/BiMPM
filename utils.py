import tensorflow as tf
import numpy as np

emb_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
fc_layer = tf.contrib.layers.fully_connected

def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def inner_production(x, y):
    return tf.reduce_sum(x * y, axis=1) + 1e-9

def cosine_distance(x, y):
    return inner_production(x, y) / tf.sqrt(inner_production(x, x) * inner_production(y, y))
