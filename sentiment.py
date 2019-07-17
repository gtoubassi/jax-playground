import nn
import jax.numpy as np
from jax import random
import numpy as onp
import json
import gzip
from random import Random
import time

class SentimentDataset:
  def __init__(self):
    npz = np.load('data/100k_reviews_binary_10k_vocab.npz', allow_pickle=True)
    num_reviews = npz['x_train'].shape[0] + npz['x_test'].shape[0]
    print("Number of reviews: %d,  vocab size: %d" % (num_reviews, npz['x_train'].shape[1]))    
    self.data = (npz['x_train'], npz['y_train'], npz['x_test'], npz['y_test'])
  
class SentimentBase(nn.NeuralNet):

  def set_dataset(self, dataset):
    self.dataset = dataset

  def load_data(self):
    return dataset.data

  def vocab_size(self):
    return self.x_train.shape[1]

  def loss(self, params, x, y):
    y_ = self.forward(params, x)
    return nn.binary_cross_entropy_loss(y_, y)

  def eval_metrics(self, y_, y):
    y_ = y_ > .5
    return nn.binary_categorical_metrics(y_, y)

class SentimentLinear(SentimentBase):
  def init_params(self):
    return [.001 * random.normal(self.rnd_key, (self.vocab_size(),)),
      np.zeros((1,))]
        
  def forward(self, params, x):
    W1, b1 = params
    # the [0] is to make sure we return a scalar.
    return nn.sigmoid(np.matmul(W1, x) + b1[0])

class Sentiment3LayerMLP(SentimentBase):
  def init_params(self):
    return [.001 * random.normal(self.rnd_key, (20, self.vocab_size())), np.zeros((20,)),
            .001 * random.normal(self.rnd_key, (20,)), np.zeros((1,))]
        
  def forward(self, params, x):
    W1, b1, W2, b2 = params
    # the [0] is to make sure we return a scalar.
    h1 = np.tanh(np.matmul(W1, x) + b1)
    return nn.sigmoid(np.matmul(W2, h1) + b2[0])

dataset = SentimentDataset()
net = SentimentLinear(optimizer=nn.GradientDescentOptimizer(learning_rate=.3))
net.set_dataset(dataset)
print("Training...")
net.train(num_epochs=10, batch_size=64)

#for lr in (.01, .03, .1, .3, 1):
#  net = Sentiment3LayerMLP(optimizer=nn.GradientDescentOptimizer(learning_rate=lr))
#  print("Training... (%s) learning_rate = %g" % (type(net).__name__, lr))
#  net.train(num_epochs=30, batch_size=64)

#for lr in (.01, .03, .1, .3, 1):
#  net = SentimentLinear(optimizer=nn.GradientDescentOptimizer(learning_rate=lr))
#  print("Training... (%s) learning_rate = %g" % (type(net).__name__, lr))
#  net.train(num_epochs=30, batch_size=64)

