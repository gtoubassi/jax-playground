import nn
import jax.numpy as np
from jax import random
from random import Random

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
    return self.dataset.data

  def vocab_size(self):
    return self.x_train.shape[1]

  def loss(self, params, x, y):
    y_ = self.forward(params, x)
    return nn.binary_cross_entropy_loss(y_, y)

  def eval_metrics(self, y_, y):
    y_ = y_ > .5
    return nn.binary_categorical_metrics(y_, y)

class SentimentLinear(SentimentBase):
  def __init__(self, initialization_scale=.001):
    self.initialization_scale = initialization_scale
    super().__init__()

  def init_params(self):
    params = [random.normal(self.rnd_key, (self.vocab_size(),)), np.zeros((1,))]
    return [self.initialization_scale * p for p in params]
      
        
  def forward(self, params, x):
    W1, b1 = params
    # the [0] is to make sure we return a scalar.
    return nn.sigmoid(np.matmul(W1, x) + b1[0])

class Sentiment3LayerMLP(SentimentBase):
  def __init__(self, initialization_scale=.001, hidden_units=20):
    self.initialization_scale = initialization_scale
    self.hidden_units = hidden_units
    super().__init__()

  def init_params(self):
    h = self.hidden_units
    params = [random.normal(self.rnd_key, (h, self.vocab_size())), np.zeros((h,)),
              random.normal(self.rnd_key, (h,)), np.zeros((1,))]
    return [self.initialization_scale * p for p in params]

  def forward(self, params, x):
    W1, b1, W2, b2 = params
    # the [0] is to make sure we return a scalar.
    h1 = np.tanh(np.matmul(W1, x) + b1)
    return nn.sigmoid(np.matmul(W2, h1) + b2[0])


def hyperparam_tune(dataset):
  run_count = 0
  rand = Random(12345)

  while True:
    run_count += 1
  
    hyperparams = {}

    hidden_units = rand.randint(10, 30)
    hyperparams['hidden_units'] = hidden_units
    initialization_scale = rand.choice([.001, .01, .1])
    hyperparams['initialization_scale'] = initialization_scale
    batch_size = rand.choice([32, 64, 128])
    hyperparams['batch_size'] = batch_size
  
    if rand.random() < .5:
      net = SentimentLinear(initialization_scale)
    else:
      net = Sentiment3LayerMLP(initialization_scale, hidden_units)
    hyperparams['model'] = net.__class__.__name__
    
    net.set_dataset(dataset)

    learning_rate = rand.uniform(.01, 3) #Maybe should sample log so we explore 0.1-1 more?
    hyperparams['learning_rate'] = learning_rate

    if rand.random() < .5:
      optimizer = nn.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
      optimizer = nn.MomentumOptimizer(learning_rate=learning_rate)
    net.set_optimizer(optimizer)
  
    hyperparams['optimizer'] = optimizer.__class__.__name__
    hyperparams_msg = ', '.join(['%s: %s' % (k, str(hyperparams[k])) for k in hyperparams])
  
    print("Starting run %d  %s" % (run_count, hyperparams_msg))
    metrics = net.train(num_epochs=20, batch_size=batch_size)
  
    metrics_msg = ', '.join(['%s: %g' % (k, metrics[k]) for k in metrics])
  
    print("Finished run %d  %s  %s" % (run_count, hyperparams_msg, metrics_msg))


dataset = SentimentDataset()

#net = SentimentLinear(.01)
#net.set_optimizer(nn.GradientDescentOptimizer(learning_rate=.5))
#net.set_dataset(dataset)
#net.train(num_epochs=20, batch_size=32)

net = Sentiment3LayerMLP(.01, 12)
net.set_optimizer(nn.GradientDescentOptimizer(learning_rate=.1))
net.set_dataset(dataset)
net.train(num_epochs=20, batch_size=64)


