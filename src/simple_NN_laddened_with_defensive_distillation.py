import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

x_adversarial = np.load('../generated_adversarial_data/fgsm/xadv.npy')
x_legitimate = np.load('../generated_adversarial_data/fgsm/xtest.npy')
data_labels = np.load('../generated_adversarial_data/fgsm/ytest.npy')

'''
x_adversarial = tf.constant(x_adversarial, tf.float32)
x_legitimate = tf.constant(x_legitimate, tf.float32)
data_labels = tf.constant(data_labels, tf.float32)
'''

x_adversarial = np.reshape(np.array(x_adversarial), (10000, 784))
x_legitimate = np.reshape(np.array(x_legitimate), (10000, 784))
data_labels = np.array(data_labels)


TEMPERATURE = 40.0
tf_temperature = tf.constant(TEMPERATURE, tf.float32)

def main(_):
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	
	### The protective network
	x_network_1 = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	W_network_1 = tf.Variable(tf.zeros([784, 10]))
	b_network_1 = tf.Variable(tf.zeros([10]))
	y_network_1 = tf.div((tf.matmul(x_network_1, W_network_1) + b_network_1), tf_temperature)

	cross_entropy_network_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_network_1))
	train_step_network_1 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_network_1)

	sess = tf.InteractiveSession()

	tf.global_variables_initializer().run()

	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step_network_1, feed_dict={x_network_1: batch_xs, y_: batch_ys})

	# Test trained model
	correct_prediction_network_1 = tf.equal(tf.argmax(y_network_1, 1), tf.argmax(y_, 1))
	accuracy_network_1 = tf.reduce_mean(tf.cast(correct_prediction_network_1, tf.float32))
	
	### The prediction network or network to be protected
	x_network_2 = tf.placeholder(tf.float32, [None, 784])
	y_network_2 = tf.placeholder(tf.float32, [None, 10])

	W_network_2 = tf.Variable(tf.zeros([784, 10]))
	b_network_2 = tf.Variable(tf.zeros([10]))
	y = tf.div((tf.matmul(x_network_2, W_network_2) + b_network_2), tf_temperature)

	cross_entropy_network_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_network_2, logits=y))
	train_step_network_2 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_network_2)

	tf.global_variables_initializer().run()

	sess.run(train_step_network_2, feed_dict={x_network_2: mnist.train.images, y_network_2: mnist.train.labels})

	# Test trained model
	correct_prediction_network_2 = tf.equal(tf.argmax(y_network_2, 1), tf.argmax(y, 1))
	accuracy_network_2 = tf.reduce_mean(tf.cast(correct_prediction_network_2, tf.float32))
	#print("Accuracy on the legitimate examples are", str(sess.run(accuracy_network_2, feed_dict={x_network_2: x_legitimate, y_network_2: data_labels})))
	print("Accuracy on the adversarial examples are", str(sess.run(accuracy_network_2, feed_dict={x_network_2: x_adversarial, y_network_2: data_labels})))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
