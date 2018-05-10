import sys, os, math

import numpy as np
import tensorflow as tf

from utils import *

class Model(object):
    """docstring for Model"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.build_model()

    def calc_max_state(self, output):
        return tf.reduce_max(output, axis=1)

    def calc_mean_state(self, output, mask):
        return tf.reduce_sum(tf.tile(tf.expand_dims(mask, axis=2), [1, 1, 2 * self.config.hidden_size]) * output, axis=1) / tf.tile(tf.reduce_sum(mask, axis=1, keep_dims=True), [1, 2 * self.config.hidden_size])

    def build_self_attention(self, x_output, x_mask, y_output, y_mask, scope='', attention_size=None):
        if attention_size is None:
            attention_size = 2 * self.config.hidden_size
        with tf.variable_scope('attention_%s' % (scope)):
            W = tf.get_variable('W_%s' % (scope), [attention_size, attention_size], tf.float32)
            U = tf.get_variable('U_%s' % (scope), [attention_size, attention_size], tf.float32)
            V = tf.get_variable('V_%s' % (scope), [attention_size, 1], tf.float32)

            temp_W = tf.einsum('aij,jk->aik', x_output, W) # batch * length * hidden_size
            temp_U = tf.einsum('aij,jk->aik', y_output, U) # batch * length * hidden_size
            temp = tf.tanh(tf.tile(tf.expand_dims(temp_U, axis=1), [1, self.config.max_length, 1, 1]) + tf.tile(tf.expand_dims(temp_W, axis=2), [1, 1, self.config.max_length, 1]))
            x_e = tf.tile(tf.expand_dims(y_mask, axis=1), [1, self.config.max_length, 1]) * tf.tile(tf.expand_dims(x_mask, axis=2), [1, 1, self.config.max_length]) * tf.exp(tf.einsum('abij,jk->abik', temp, V)[:, :, :, 0]) # batch * length * length
            _x_a = x_e / tf.tile(tf.reduce_sum(x_e, axis=2, keep_dims=True), [1, 1, self.config.max_length]) # batch * length
            x_a = tf.tile(tf.expand_dims(_x_a, axis=3), [1, 1, 1, attention_size]) # batch * length * length * 2 hidden_size
            attention_x_output = tf.reduce_sum(tf.tile(tf.expand_dims(y_output, axis=1), [1, self.config.max_length, 1, 1]) * x_a, axis=2)

            temp_W = tf.einsum('aij,jk->aik', y_output, W) # batch * length * hidden_size
            temp_U = tf.einsum('aij,jk->aik', x_output, U) # batch * length * hidden_size
            temp = tf.tanh(tf.tile(tf.expand_dims(temp_U, axis=1), [1, self.config.max_length, 1, 1]) + tf.tile(tf.expand_dims(temp_W, axis=2), [1, 1, self.config.max_length, 1]))
            y_e = tf.tile(tf.expand_dims(x_mask, axis=1), [1, self.config.max_length, 1]) * tf.tile(tf.expand_dims(y_mask, axis=2), [1, 1, self.config.max_length]) * tf.exp(tf.einsum('abij,jk->abik', temp, V)[:, :, :, 0]) # batch * length * length
            _y_a = y_e / tf.tile(tf.reduce_sum(y_e, axis=2, keep_dims=True), [1, 1, self.config.max_length]) # batch * length
            y_a = tf.tile(tf.expand_dims(_y_a, axis=3), [1, 1, 1, attention_size]) # batch * length * length * 2 hidden_size
            attention_y_output = tf.reduce_sum(tf.tile(tf.expand_dims(x_output, axis=1), [1, self.config.max_length, 1, 1]) * y_a, axis=2) # batch * length * 2 hidden_size

        return attention_x_output, attention_y_output

    def build_model(self):
        self.input_x_placeholder = tf.placeholder(tf.int32, [None, self.config.max_length], 'input_x')
        self.input_x_length_placeholder = tf.placeholder(tf.int32, [None], 'input_x_length')
        self.input_y_placeholder = tf.placeholder(tf.int32, [None, self.config.max_length], 'input_y')
        self.input_y_length_placeholder = tf.placeholder(tf.int32, [None], 'input_y_length')
        self.label_placeholder = tf.placeholder(tf.int32, [None], 'label')
        self.placeholder_list = [self.input_x_placeholder, self.input_x_length_placeholder, self.input_y_placeholder, self.input_y_length_placeholder, self.label_placeholder]

        embed = tf.get_variable('embed', [self.config.vocab_size, self.config.embedding_size], tf.float32)
        emb_x = tf.nn.embedding_lookup(embed, self.input_x_placeholder)
        emb_y = tf.nn.embedding_lookup(embed, self.input_y_placeholder)
        x_mask = tf.sequence_mask(self.input_x_length_placeholder, maxlen=self.config.max_length, dtype=tf.float32)
        y_mask = tf.sequence_mask(self.input_y_length_placeholder, maxlen=self.config.max_length, dtype=tf.float32)
        one_hot_label = tf.one_hot(self.label_placeholder, depth=self.config.num_class, on_value=1., off_value=0., dtype=tf.float32)

        for i in range(self.config.communication_layer_num):
            fw_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            bw_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            if i == 0:
                input_x = emb_x
                input_y = emb_y
            with tf.variable_scope('encoder_%d' % (i)):
                x_output, x_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_x,
                        self.input_x_length_placeholder, dtype=tf.float32)
                x_output = tf.concat(x_output, axis=2) # batch * length * 2 hidden_size
                x_state = tf.concat([x[0] for x in x_state], axis=1) # batch * 2 hidden_size

            with tf.variable_scope('encoder_%d' % (i), reuse=True):
                y_output, y_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_y,
                        self.input_y_length_placeholder, dtype=tf.float32)
                y_output = tf.concat(y_output, axis=2)
                y_state = tf.concat([x[0] for x in y_state], axis=1) # batch * 2 hidden_size

            if i == 0:
	            x_state, y_state, _, _ = self.build_self_attention(x_state, x_output, x_mask, y_state, y_output, y_mask, scope='%d' % (i))

	            input_x = tf.concat([emb_x, x_state], axis=2) # batch * length * 3 hidden_size
	            input_y = tf.concat([emb_y, y_state], axis=2) # batch * length * 3 hidden_size
            if i == self.config.communication_layer_num - 1:
                end_state = [x_state, y_state]

        feature = tf.concat(end_state, axis=1) # batch * 4 hidden_size

        with tf.variable_scope('matching'):
            feature = fc_layer(feature, self.config.hidden_size, activation_fn=tf.nn.relu) # batch * hidden_size
            feature = fc_layer(feature, 64, activation_fn=tf.nn.relu) # batch * 64
            feature = fc_layer(feature, self.config.num_class, activation_fn=None) # batch * 2

        with tf.variable_scope('loss'):
            self.loss = tf.losses.softmax_cross_entropy(one_hot_label, feature, reduction=tf.losses.Reduction.MEAN)
            self.prediction = tf.cast(tf.nn.softmax(feature)[:, 1] > 0.5, dtype=tf.int32)
            self.right = tf.cast(tf.equal(self.prediction, self.label_placeholder), dtype=tf.float32)

            loss = self.loss

        if self.config.optimizer == 'Adam':
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
        elif self.config.optimizer == 'SGD':
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step, 10000, 0.96, staircase=False)

            weight = tf.cast(self.global_step, dtype=tf.float32) / (tf.cast(self.global_step, dtype=tf.float32) + 50000.)
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss, global_step=self.global_step)

    def make_feed_dict(self, data):
        feed_dict = {}
        for x, y in zip(self.placeholder_list, data):
            feed_dict[x] = y
        return feed_dict, data[-1].shape[0]

    def get_infer_data(self, feed_dict):
        return feed_dict[self.input_x_placeholder], feed_dict[self.input_x_length_placeholder], feed_dict[self.input_y_placeholder], feed_dict[self.input_y_length_placeholder], feed_dict[self.label_placeholder]
