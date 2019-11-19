import os
import sys
sys.path.append('..') 
import tensorflow as tf

class Entity():
    def __init__(self, data):
        self.y = data[0]     # MNIST教師データ
        self.x = data[1:785] # MNIST画素データ