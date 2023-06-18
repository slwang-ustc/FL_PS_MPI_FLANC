class ClientConfig:
    def __init__(self, idx):
        self.idx = idx
        # self.params = None
        self.params_dict = None
        self.epoch_idx = 0

        self.train_data_idxes = None
        self.model_ratio = 1.0
        # self.model_type = None
        # self.dataset_type = None
        # self.batch_size = None
        self.lr = None
        # self.decay_rate = None
        # self.min_lr = None
        # self.epoch = None
        # self.momentum = None
        # self.weight_decay = None
        # self.local_steps = 20

        self.train_time = 0
        self.send_time = 0
