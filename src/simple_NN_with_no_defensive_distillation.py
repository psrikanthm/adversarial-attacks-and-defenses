import numpy as np
import argparse
import sys

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

def main(_):
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.matmul(x, W) + b

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()

	tf.global_variables_initializer().run()

	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Accuracy on the legitimate examples are", str(sess.run(accuracy, feed_dict={x: x_legitimate, y_: data_labels})))
	#print("Accuracy on the adversarial examples are", str(sess.run(accuracy, feed_dict={x: x_adversarial, y_: data_labels})))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
