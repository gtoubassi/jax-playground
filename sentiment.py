import nn
import jax.numpy as np
import numpy as onp
import json
import gzip
from random import Random
from jax import random

class SentimentNet(nn.NeuralNet):

  def load_data(self):
    with gzip.GzipFile('data/50k_reviews10k.json.gz', 'r') as fp:
      topk, reviews = json.load(fp)

    # Remove 3 star reviews to simplify problem
    reviews = list(filter(lambda r: r[1] != 3.0, reviews))

    num_positive = sum(1 for r in reviews if r[1] > 3)
    num_negative = len(reviews) - num_positive
    
    if False:
      # Balance out the dataset so we have equal positive/negative
      assert num_negative < num_positive
      # sort from negative to positive
      reviews.sort(key = lambda r:r[1])
      # grab the least favorable and the most favorable equally
      reviews = reviews[0:num_negative] + reviews[-num_negative:]
      num_positive = sum(1 for r in reviews if r[1] > 3)
      num_negative = len(reviews) - num_positive

    print("Total of %d reviews (%d%% positive)" % (len(reviews), num_positive * 100 / len(reviews)))

    Random(12345).shuffle(reviews)
    self.vocab_size = len(topk)
    ident_topk = np.eye(self.vocab_size)
    x_all = onp.zeros((len(reviews), self.vocab_size))
    y_all = onp.zeros((len(reviews),))
    for i, r in enumerate(reviews):
      for w in r[0]:
        x_all[i, w] = 1
      # Make it a binary classification (positive/negative review)
      y_all[i] = 1.0 if r[1] > 3 else 0.0
  
    return (x_all[0:int(len(reviews)*.9)],
            y_all[0:int(len(reviews)*.9)],
            x_all[int(len(reviews)*.9):],
            y_all[int(len(reviews)*.9):])

  def init_params(self):
    return [.001 * random.normal(self.rnd_key, (self.vocab_size,)),
      np.zeros((1,))]
        
  def forward(self, params, x):
    W1, b1 = params
    # the [0] is to make sure we return a scalar.
    return nn.sigmoid(np.matmul(W1, x) + b1[0])

  def loss(self, params, x, y):
    y_ = self.forward(params, x)
    return nn.binary_cross_entropy_loss(y_, y)

  def eval_metrics(self, y_, y):
    y_ = y_ > .5
    return nn.binary_categorical_metrics(y_, y)

net = SentimentNet()
print("Training...")
net.train(num_epochs=10, learning_rate=.1, batch_size=64)

