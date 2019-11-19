import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.utils import shuffle


class LeNet(Model):
    def __init__(self):
        super().__init__()
        '''
        Sub Model
        '''
        self.Convolution = LeNetConv()
        self.FullConnection = LeNetFC()
        
        '''
        損失関数定義
        '''
        self.criterion = tf.losses.CategoricalCrossentropy()
        
        '''
        Optimizer定義
        '''
        self.optimizer = tf.keras.optimizers.Adam()
        
        '''
        Loss(損失)やAcc(精度)の記録オブジェクト定義
        '''
        self.train_loss = tf.keras.metrics.Mean()
        self.train_acc = tf.keras.metrics.CategoricalAccuracy()
        self.valid_loss = tf.keras.metrics.Mean()
        self.valid_acc = tf.keras.metrics.CategoricalAccuracy()

    def call(self, x):
        z = self.Convolution(x)
        y = self.FullConnection(z)
        
        return y
    
    @tf.function
    def compute_loss(self, label, pred):
        return self.criterion(label, pred)

    @tf.function
    def train_step(self, x, t):
        with tf.GradientTape() as tape:
            preds = self.call(x)
            loss = self.compute_loss(t, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.train_loss(loss)
        self.train_acc(t, preds)

        return preds

    @tf.function
    def valid_step(self, x, t):
        preds = self.call(x)
        loss = self.compute_loss(t, preds)
        self.valid_loss(loss)
        self.valid_acc(t, preds)

        return preds
            
class LeNetConv(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(6, kernel_size=(5, 5),
                            padding='valid', activation='relu')
        self.pooling1 = MaxPooling2D(padding='same')
        self.conv2 = Conv2D(16, kernel_size=(5, 5),
                            padding='valid', activation='relu')
        self.pooling2 = MaxPooling2D(padding='same')


    def call(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        y = self.pooling2(x)

        return y
    
class LeNetFC(Model):
    def __init__(self):
        super().__init__()
        self.flat = Flatten()
        self.fc1 = Dense(120, activation='relu')
        self.fc2 = Dense(84, activation='relu')
        self.out = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.out(x)

        return y