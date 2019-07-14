import jax.numpy as np
from jax import grad, value_and_grad, jit, vmap
from jax import random
import tensorflow as tf
import numpy as onp

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.save('mnist.npy', (x_train, y_train, x_test, y_test))
