import os
import sys

ROOT_PATH = '/home/jovyan/work'
PROJECT_NAME = os.path.dirname(os.getcwd()).split("/")[-1]

class Config():
    def __init__(self):
        self.seed = 1234
        self.model = Model()
        self.tfboard = TFB()
        self.dataset = Dataset()
    
class Model():
    def __init__(self):
        self.max_epoch = 10
        self.batch_size = 64
        self.input_shape = [32, 32, 3] # [h, w, c]
        
        self.fc1_unit_num = 120
        self.fc2_unit_num = 84
        self.output_unit_num = 10
        
        self.checkpoint_dir = os.path.join(ROOT_PATH, '_checkpoint', PROJECT_NAME)
        if not os.path.isdir(self.checkpoint_dir):
            print('{} is not found. so, created.'.format(self.checkpoint_dir))
            os.makedirs(self.checkpoint_dir)

class TFB():
    def __init__(self):
        self.train_log_dir = os.path.join(ROOT_PATH, '_log', PROJECT_NAME, 'train')
        if not os.path.isdir(self.train_log_dir):
            print('{} is not found. so, created.'.format(self.train_log_dir))
            os.makedirs(self.train_log_dir)

        self.valid_log_dir = os.path.join(ROOT_PATH, '_log', PROJECT_NAME, 'valid')
        if not os.path.isdir(self.valid_log_dir):
            print('{} is not found. so, created.'.format(self.valid_log_dir))
            os.makedirs(self.valid_log_dir)

class Dataset():
    def __init__(self):
        '''オリジナルデータセットのパス'''
        self.train_csv_path = os.path.join(ROOT_PATH, '_dataset', 'mnist', 'mnist_train.csv')
        self.valid_csv_path = os.path.join(ROOT_PATH, '_dataset', 'mnist', 'mnist_test.csv')
        
        '''仮データセットのパス
        
        Remark:
        仮データセットがマウント先にある時、
        オリジナルデータに手を加えたくないとき
        など、
        速度改善にこのパスを使ったりするとGood.
        '''
        # self.tmp_path = os.path.join(ROOT_PATH, 'TMP')
        # self.tmp_path = os.path.join(ROOT_PATH, 'DATASET')
        # if not os.path.isdir(self.tmp_path):
        #     print('{} is not found. so, created.'.format(self.tmp_path))
        #     os.makedirs(self.tmp_path)
        