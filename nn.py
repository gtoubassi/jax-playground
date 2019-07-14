import jax.numpy as np
from jax import grad, value_and_grad, jit, vmap
from jax import random
from functools import partial

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def softmax(x):
  e_x = np.exp(x - np.max(x)) # for stability see blog.feedly.com/tricks-of-the-trade-logsumexp/
  return e_x / np.sum(e_x)

def cross_entropy_loss(y_, y):
  assert len(y_.shape) == len(y.shape)
  return -np.sum(y * np.log(y_ + 1e-8))

def binary_cross_entropy_loss(y_, y):
  assert len(y_.shape) == len(y.shape)
  return -(y * np.log(y_ + 1e-8) + (1-y) * np.log((1-y_) + 1e-8))

def mse_loss(y_, y):
  assert len(y_.shape) == len(y.shape)
  return np.mean((y - y_)**2)

def onehot_multi_categorical_accuracy(y_, y):
  assert len(y_.shape) == len(y.shape)
  # y and y_ are batches of one hot encoded predictions, targets
  # if y and y_ for an example don't match then y*y_ will be a vector of zeros,
  # if they do match a one hot encoded vector.  So sum will render a 1 for match,
  # 0 for mismatch
  return np.mean(np.sum(y_ * y, axis=1))

def binary_categorical_accuracy(y_, y):
  assert len(y_.shape) == len(y.shape)
  return np.mean(y_ == y)

def binary_categorical_metrics(y_, y):
  assert len(y_.shape) == len(y.shape)
  accuracy = np.mean(y_ == y)

  true_positives = np.dot(y_, y)
  false_positives = np.dot(y_, 1-y)
  false_negatives = np.dot(1-y_, y)

  precision = true_positives / (true_positives + false_positives)
  recall = true_positives / (true_positives + false_negatives)
  f1 = 2 * precision * recall / (precision + recall)
  return {'accuracy': accuracy, 'F1': f1, 'precision': precision, 'recall': recall}


class NeuralNet:
  def __init__(self):
    self.val_and_grad = jit(vmap(value_and_grad(partial(self.__class__.loss, self)), in_axes=(None, 0, 0)))
    self.vforward = jit(vmap(partial(self.__class__.forward, self), in_axes=(None, 0)))
    self.vloss = jit(vmap(partial(self.__class__.loss, self), in_axes=(None, 0, 0)))
    self.rnd_key = random.PRNGKey(0)
    (self.x_train, self.y_train, self.x_test, self.y_test) = self.load_data()
    self.params = self.init_params()
  
  def accumulate_gradients(self, batch_x, batch_y):
    batch_size = batch_x.shape[0]
    (v, g) = self.val_and_grad(self.params, batch_x, batch_y)
    ave_g = [np.mean(x, axis=0) for x in g]
    ave_v = np.mean(v)    
    return (ave_v, ave_g)

  def train_batch(self, learning_rate, batch_x, batch_y):
    (v, g) = self.accumulate_gradients(batch_x, batch_y)
    self.params = [p - g[idx]*learning_rate for idx, p in enumerate(self.params)]
    return v

  def train_epoch(self, learning_rate, batch_size):
    num_batches = 0
    accum_loss = 0
    for b in range(0, self.x_train.shape[0], batch_size):
      batch_x = self.x_train[b:(b + batch_size)]
      batch_y = self.y_train[b:(b + batch_size)]
      
      accum_loss += self.train_batch(learning_rate, batch_x, batch_y)
      num_batches += 1
    print("Ave loss over epoch", (accum_loss / num_batches))

  def train(self, num_epochs=10, learning_rate = .1, batch_size = 64):
    #self.log_eval('Initial train set', self.x_train, self.y_train)
    self.log_eval('Initial test set', self.x_test, self.y_test)
    for i in range(num_epochs):
      print('epoch', i)
      self.train_epoch(learning_rate, batch_size)
      #self.log_eval('Train set', self.x_train, self.y_train)
      self.log_eval('Test set', self.x_test, self.y_test)

  def log_eval(self, msg, x, y):
    metrics = self.eval_metrics(self.vforward(self.params, x), y)
    print('[%s] ' % msg, end='')
    metrics_msg = ', '.join(['%s: %g' % (k, metrics[k]) for k in metrics])
    print(metrics_msg)

  def load_data(self):
    raise NotImplementedError("abstract")

  def init_params(self):
    raise NotImplementedError("abstract")

  def forward(self, params, x):
    raise NotImplementedError("abstract")

  def loss(self, params, x, y):
    raise NotImplementedError("abstract")

  def eval_metrics(self, y_, y):
    raise NotImplementedError("abstract")


