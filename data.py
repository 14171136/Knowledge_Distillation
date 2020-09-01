# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:53:00 2020

@author: riyuecao
"""
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

class Dataset:
    def __init__(self,args):
        self.mnist = input_data.read_data_sets('./data/mnist',one_hot=True)
        tf.logging.set_verbosity(old_v)
        self.num_samples = self.mnist.train.num_examples
        self.batch_size = args.batch_size
        self.num_batches = int(self.num_samples/self.batch_size)
        
    def get_test_data(self):
        return (self.mnist.test.images, self.mnist.test.labels)
    
    def get_train_data(self):
        return self.mnist.train
    
    def get_validation_data(self):
        return (self.mnist.validation.images, self.mnist.validation.labels)

