import random
import sys
import os
import time

import numpy as np
import tensorflow as tf

import Config
import Model
import Data

random.seed(19960322)

config = Config.Config()
data = Data.Data(config)
model = Model.Model(config)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    print('begin init...')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    if config.use_latest_params:
        print('load params from %s' % (config.checkpoints))
        saver.restore(sess, tf.train.latest_checkpoint(config.checkpoints))
    else:
        print('initialize with fresh params...')
        sess.run(tf.global_variables_initializer())
    print('finish init.')

    right = []
    total = []
    loss = []
    distribution_loss = []
    std_loss = []
    for item in data.get_dev_data():
        feed_dict, _total = model.make_feed_dict(item)
        _loss, _distribution_loss, _std_loss, _right, _sim_x_a, _sim_y_a, _diff_x_a, _diff_y_a, _sim_result, _diff_result, _prediction = sess.run([model.loss, model.distribution_loss, model.std_loss, model.right, \
                                                                model.sim_x_a, model.sim_y_a, model.diff_x_a, model.diff_y_a, model.sim_result, model.diff_result, model.prediction], feed_dict=feed_dict)
        right.append(_right)
        total.append(_total)
        loss.append(_loss)
        distribution_loss.append(_distribution_loss)
        std_loss.append(_std_loss)
        x, x_len, y, y_len, label = model.get_infer_data(feed_dict)
        data.print_infer(x, x_len, y, y_len, label, _sim_x_a, _sim_y_a, _diff_x_a, _diff_y_a, _sim_result, _diff_result, _prediction)
    print('dev---loss: %f, distribution loss: %f, std loss: %f, precise: %f' % (np.mean(loss), np.mean(distribution_loss), np.mean(std_loss), np.sum(right) / np.sum(total)))

    right = []
    total = []
    loss = []
    distribution_loss = []
    std_loss = []
    for item in data.get_test_data():
        feed_dict, _total = model.make_feed_dict(item)
        _loss, _distribution_loss, _std_loss, _right, _sim_x_a, _sim_y_a, _diff_x_a, _diff_y_a, _sim_result, _diff_result, _prediction = sess.run([model.loss, model.distribution_loss, model.std_loss, model.right, \
                                                                model.sim_x_a, model.sim_y_a, model.diff_x_a, model.diff_y_a, model.sim_result, model.diff_result, model.prediction], feed_dict=feed_dict)
        right.append(_right)
        total.append(_total)
        loss.append(_loss)
        distribution_loss.append(_distribution_loss)
        std_loss.append(_std_loss)
        x, x_len, y, y_len, label = model.get_infer_data(feed_dict)
        data.print_infer(x, x_len, y, y_len, label, _sim_x_a, _sim_y_a, _diff_x_a, _diff_y_a, _sim_result, _diff_result, _prediction)
    print('test---loss: %f, distribution loss: %f, std loss: %f, precise: %f' % (np.mean(loss), np.mean(distribution_loss), np.mean(std_loss), np.sum(right) / np.sum(total)))
