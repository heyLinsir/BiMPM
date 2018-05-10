class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        self.learning_rate = 1e-4
        self.optimizer = 'Adam' # Adam or Adagrad or Adadelta or SGD

        self.hidden_size = 300
        self.embedding_size = 300
        self.num_class = 2
        self.max_length = 30
        self.keep_prob = 0.5
        self.communication_layer_num = 2

        self.batch_size = 64
        self.epoch = 50
        self.checkpoints = './checkpoints'
        self.use_latest_params = False

        self.data_path = './data-bimpm'
        self.train_file = 'train.pkl'
        self.test_file = 'test.pkl'
        self.dev_file = 'dev.pkl'
        self.word2id = 'word2id.pkl'
        self.id2word = 'id2word.pkl'
