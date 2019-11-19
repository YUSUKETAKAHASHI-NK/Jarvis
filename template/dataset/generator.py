import numpy as np
from sklearn.utils import shuffle

class Generator():
    def __init__(self, dataset, batch_size=32, shuffle=False, random_state=1234):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if random_state is None:
            random_state = np.random.RandomState(1234)
        self.random_state = random_state
        self._idx = 0
        self._reset()
        
    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N //b + bool(N%b)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()
        
        data_list = self.dataset[self._idx:(self._idx + self.batch_size)]
        x_list = []
        y_list = []
        
        for data in data_list:
            x_list.append((data.x.reshape(28, 28, 1) / 255).astype(np.float32))
            y_list.append(np.eye(10)[data.y].astype(np.float32))
        
        batch_x = np.array(x_list)
        batch_y = np.array(y_list)
        
        self._idx += self.batch_size
        
        return batch_x, batch_y
    
    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset, random_state = self.random_state)
        self._idx = 0
        