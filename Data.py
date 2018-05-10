import random
import pickle as pickle

import numpy as np

class Data(object):
    """docstring for Data"""
    def __init__(self, config):
        super(Data, self).__init__()
        self.config = config

        def padding(sen, pad_id, max_length):
            if len(sen) >= max_length:
                return sen[:max_length], max_length
            length = len(sen)
            return sen + [pad_id] * (max_length - length), length

        def postprocess(data, pad_id, max_length, is_training=False):
            _data = []
            for item in data:
                sen_x, x_length = padding(item['sen_x'], pad_id, max_length)
                sen_y, y_length = padding(item['sen_y'], pad_id, max_length)
                _data.append({'x': sen_x,
                                'x_length': x_length,
                                'y': sen_y,
                                'y_length': y_length,
                                'label': item['label']})
                if is_training and item['label'] == 1:
                    _data.append({'x': sen_x,
                                    'x_length': x_length,
                                    'y': sen_y,
                                    'y_length': y_length,
                                    'label': item['label']})
            return _data

        print('begin load data')
        self.word2id = pickle.load(open('%s/%s' % (config.data_path, config.word2id), 'rb'))
        self.id2word = pickle.load(open('%s/%s' % (config.data_path, config.id2word), 'rb'))
        self.train_data = postprocess(pickle.load(open('%s/%s' % (config.data_path, config.train_file), 'rb')), self.word2id['<PAD>'], self.config.max_length, is_training=True)
        self.test_data = postprocess(pickle.load(open('%s/%s' % (config.data_path, config.test_file), 'rb')), self.word2id['<PAD>'], self.config.max_length)
        self.dev_data = postprocess(pickle.load(open('%s/%s' % (config.data_path, config.dev_file), 'rb')), self.word2id['<PAD>'], self.config.max_length)
        print('load finished')
        print('train data: %d' % (len(self.train_data)))
        print('test data: %d' % (len(self.test_data)))
        print('dev data: %d' % (len(self.dev_data)))
        print('vocab size: %d' % (len(self.id2word)))
        config.vocab_size = len(self.id2word)

    def id2sentence(self, x):
        return [self.id2word[word] for word in x.tolist()]

    def print_infer(self, x, x_length, y, y_length, label, sim_x_a, sim_y_a, diff_x_a, diff_y_a, sim_result, diff_result, prediction):
        def pred(score_list, acc_weight=0.85):
            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
            weight = 0.
            for i in xrange(len(score_list)):
                weight += score_list[i][1]
                if weight >= acc_weight:
                    return score_list[:i + 1]
            return score_list
        for _x, _x_length, _y, _y_length, _label, _sim_x_a, _sim_y_a, _diff_x_a, _diff_y_a, _sim_result, _diff_result, _prediction in zip(x, x_length, y, y_length, label, sim_x_a, sim_y_a, diff_x_a, diff_y_a, sim_result, diff_result, prediction):
            _x = self.id2sentence(_x)[:_x_length]
            _y = self.id2sentence(_y)[:_y_length]
            print('')
            print(' '.join(_x))

            sim_attention_list = []
            diff_attention_list = []
            for i in xrange(_x_length):
                sim_attention_list.append((_x[i], _sim_x_a[i]))
                diff_attention_list.append((_x[i], _diff_x_a[i]))
            sim_attention_list = pred(sim_attention_list)
            diff_attention_list = pred(diff_attention_list)

            sim_info = []
            diff_info = []
            for item in sim_attention_list:
                sim_info.append('%s(%f)' % (item[0], round(item[1], 4)))
            for item in diff_attention_list:
                diff_info.append('%s(%f)' % (item[0], round(item[1], 4)))
            print('\tsimilar attention---%s' % ('\t'.join(sim_info)))
            print('\tdiffere attention---%s' % ('\t'.join(diff_info)))

            print(' '.join(_y))

            sim_attention_list = []
            diff_attention_list = []
            for i in xrange(_y_length):
                sim_attention_list.append((_y[i], _sim_y_a[i]))
                diff_attention_list.append((_y[i], _diff_y_a[i]))
            sim_attention_list = pred(sim_attention_list)
            diff_attention_list = pred(diff_attention_list)

            sim_info = []
            diff_info = []
            for item in sim_attention_list:
                sim_info.append('%s(%f)' % (item[0], round(item[1], 4)))
            for item in diff_attention_list:
                diff_info.append('%s(%f)' % (item[0], round(item[1], 4)))
            print('\tsimilar attention---%s' % ('\t'.join(sim_info)))
            print('\tdiffere attention---%s' % ('\t'.join(diff_info)))

            print('ground truth: %d' % (_label))
            print('prediction:   %d' % (_prediction))
            print('similar att: %f\t%f' % (round(_sim_result[0], 4), round(_sim_result[1], 4)))
            print('differe att: %f\t%f' % (round(_diff_result[0], 4), round(_diff_result[1], 4)))

    def shuffle(self):
        random.shuffle(self.train_data)

    def get_train_data(self, batch_size=-1):
        self.shuffle()
        if batch_size < 0:
            batch_size = self.config.batch_size
        batch_cnt = int(len(self.train_data) / batch_size)
        for i in range(batch_cnt):
            data = self.train_data[i * batch_size:(i + 1) * batch_size]
            x = []
            y = []
            x_length = []
            y_length = []
            label = []
            for item in data:
                x.append(item['x'])
                y.append(item['y'])
                x_length.append(item['x_length'])
                y_length.append(item['y_length'])
                label.append(item['label'])
            x = np.asarray(x, 'int32')
            y = np.asarray(y, 'int32')
            x_length = np.asarray(x_length, 'int32')
            y_length = np.asarray(y_length, 'int32')
            label = np.asarray(label, 'int32')
            yield [x, x_length, y, y_length, label]

    def get_test_data(self, batch_size=-1):
        if batch_size < 0:
            batch_size = self.config.batch_size
        batch_cnt = int(len(self.test_data) / batch_size)
        for i in range(batch_cnt):
            data = self.test_data[i * batch_size:(i + 1) * batch_size]
            x = []
            y = []
            x_length = []
            y_length = []
            label = []
            for item in data:
                x.append(item['x'])
                y.append(item['y'])
                x_length.append(item['x_length'])
                y_length.append(item['y_length'])
                label.append(item['label'])
            x = np.asarray(x, 'int32')
            y = np.asarray(y, 'int32')
            x_length = np.asarray(x_length, 'int32')
            y_length = np.asarray(y_length, 'int32')
            label = np.asarray(label, 'int32')
            yield [x, x_length, y, y_length, label]

    def get_dev_data(self, batch_size=-1):
        if batch_size < 0:
            batch_size = self.config.batch_size
        batch_cnt = int(len(self.dev_data) / batch_size)
        for i in range(batch_cnt):
            data = self.dev_data[i * batch_size:(i + 1) * batch_size]
            x = []
            y = []
            x_length = []
            y_length = []
            label = []
            for item in data:
                x.append(item['x'])
                y.append(item['y'])
                x_length.append(item['x_length'])
                y_length.append(item['y_length'])
                label.append(item['label'])
            x = np.asarray(x, 'int32')
            y = np.asarray(y, 'int32')
            x_length = np.asarray(x_length, 'int32')
            y_length = np.asarray(y_length, 'int32')
            label = np.asarray(label, 'int32')
            yield [x, x_length, y, y_length, label]
