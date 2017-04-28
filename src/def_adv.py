from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

def get_yarray(size,Y):
    (m,_) = size
    arr = np.zeros(size)
    for i in range(m):
        arr[i,Y[i]] = 1
    return arr

def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Redefine TF model graph
    model_2 = cnn_model()
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.2)
    predictions_2_adv = model_2(adv_x_2)

    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }

    ytest_fgsm = np.load("fgsm/ytest.npy")
    xadv_fgsm = np.load("fgsm/xadv.npy")
    xtest_fgsm = np.load("fgsm/xtest.npy")

    xtest_black = np.load("black/xtest.npy")
    ytest_black = np.load("black/ytest.npy")
    xadv_black = np.load("black/xadv.npy")

    xtest_jsma = np.load("jsma/xtest.npy")
    ytest_jsma = np.load("jsma/ytest.npy")
    xadv_jsma = np.load("jsma/xadv.npy")
    yadv_jsma = np.load("jsma/ytest2.npy")

    b = []
    for a in xadv_jsma:
        b.append(a[0])
    xadv_jsma = np.array(b)
    print(ytest_jsma.shape)
    print(yadv_jsma.shape)
    #ytest_jsma = get_yarray((len(ytest_jsma), 10),ytest_jsma)
    yadv_jsma = get_yarray((len(yadv_jsma), 10),yadv_jsma)

    def evaluate_2():
        # Evaluate the accuracy of the adversarialy trained MNIST model on
        # legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, X_test,
                                  Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, xtest_fgsm, ytest_fgsm,
                                                              args=eval_params)
        print('Test accuracy on adversarial examples1: ' + str(accuracy_adv))
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, xadv_fgsm, ytest_fgsm,
                                              args=eval_params)
        print('Test accuracy on adversarial examples2: ' + str(accuracy_adv))

        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, xtest_black, ytest_black,args=eval_params)
        print('Test accuracy on adversarial examples3: ' + str(accuracy_adv))
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, xadv_black, ytest_black,args=eval_params)
        print('Test accuracy on adversarial examples4: ' + str(accuracy_adv))

        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, xtest_jsma, ytest_jsma,args=eval_params)
        print('Test accuracy on adversarial examples5: ' + str(accuracy_adv))
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, xadv_jsma, yadv_jsma,args=eval_params)
        print('Test accuracy on adversarial examples6: ' + str(accuracy_adv))

    # Perform adversarial training
    model_train(sess, x, y, predictions_2, X_train, Y_train,
                predictions_adv=predictions_2_adv, evaluate=evaluate_2,
                args=train_params)

if __name__ == '__main__':
    app.run()
