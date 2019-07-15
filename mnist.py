import jax.numpy as np
import nn as nn
from jax import random

class MnistNetwork(nn.NeuralNet):  

  def load_data(self):
    npz = np.load('data/mnist.npz', allow_pickle=True)
    (x_train, y_train, x_test, y_test) = npz['mnist']
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    x_train = x_train.reshape((60000, 28*28)) / 255.0
    x_test = x_test.reshape((10000, 28*28)) / 255.0
    
    return (x_train, y_train, x_test, y_test)

  def init_params(self):
    return [.0001 * random.normal(self.rnd_key, (100, 28*28)),
      np.zeros((100,)),
      .0001 * random.normal(self.rnd_key, (10, 100)),
      np.zeros((10,))]

  def forward(self, params, x):
    W1, b1, W2, b2 = params
    #h1 = nn.sigmoid(np.matmul(W1, x) + b1)
    #h1 = np.maximum(np.matmul(W1, x) + b1, 0) #relu
    h1 = np.tanh(np.matmul(W1, x) + b1)
    return nn.softmax(np.matmul(W2, h1) + b2)

  def loss(self, params, x, y):
    y_ = self.forward(params, x)
    return nn.cross_entropy_loss(y_, y)

  def eval_metrics(self, y_, y):
    y_chosen = np.eye(10)[np.argmax(y_, axis=1)]
    return {'accuracy': nn.onehot_multi_categorical_accuracy(y_chosen, y)}

net = MnistNetwork(optimizer=nn.GradientDescentOptimizer(learning_rate=.5))
net.train(num_epochs=10)


