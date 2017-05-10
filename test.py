import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from mnist_rnn import RNN, weights, biases

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
learning_rate = 0.001
training_iters = 100000
batch_size = 1

n_inputs = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = RNN(x,weights,biases)

saver = tf.train.Saver()
checkpoint_dir = "checkpoint/"

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess,ckpt.model_checkpoint_path)

    input, _ = mnist.train.next_batch(batch_size)
    input = input.reshape(28, 28)
    # print(input.shape)

    im = Image.fromarray(np.uint8(input * 255))
    im.show()

    input = input.reshape(1, 28, 28)
    output = sess.run(y,feed_dict={x:input})
    y = tf.argmax(output,1)
    print(sess.run(y))




