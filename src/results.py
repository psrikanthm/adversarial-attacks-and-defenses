from sklearn import svm, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from distutils.version import LooseVersion
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import Activation
#from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_tf import model_train, model_eval, batch_eval

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

ytest_jsma = np.load("jsma/ytest2.npy")
ytest_jsma = ytest_jsma[:len(ytest_jsma)/4]

def initialize(prefix):
    x_train = np.load(prefix + "/xtrain.npy")
    y_train = np.load(prefix + "/ytrain.npy")
    x_test = np.load(prefix + "/xtest.npy")
    y_test = np.load(prefix + "/ytest.npy")
    x_adv = np.load(prefix + "/xadv.npy")

#########
## Temporary measure
    x_train = x_train[:len(x_train)/4]
    y_train = y_train[:len(y_train)/4]
    x_test = x_test[:len(x_test)/4]
    y_test = y_test[:len(y_test)/4]
    x_adv = x_adv[:len(x_adv)/4]

    return (x_train, y_train, x_test, x_adv, y_test)
######

def baseline_model(input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)
    
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


def reshape(x):
    n_samples = len(x)
    return x.reshape((n_samples, -1))

def reshape_y(y):
    return np.argmax(y, axis = 1)

def get_yarray(size,Y):
    (m,_) = size
    arr = np.zeros(size)
    for i in range(m):
        arr[i,Y[i]] = 1
    return arr

def run(x_train, y_train, x_test1, x_adv1, y_test1, x_test2, x_adv2, y_test2, x_test3, x_adv3, y_test3):
    '''
    #################
    # Test with logistic regression
    lr = LogisticRegression(C=1e5, n_jobs = 10) 
    lr.fit(reshape(x_train), reshape_y(y_train))
    
    y_pred1 = lr.predict(reshape(x_test1))
    y_pred2 = lr.predict(reshape(x_test2))
    y_pred3 = lr.predict(reshape(x_test3))
    fgsm_test = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    jsma_test = metrics.accuracy_score(reshape_y(y_test2), y_pred2)
    black_test = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    y_pred1 = lr.predict(reshape(x_adv1))
    y_pred2 = lr.predict(reshape(x_adv2))
    y_pred3 = lr.predict(reshape(x_adv3))
    fgsm_adv = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    print x_adv2.shape, y_pred2.shape, ytest_jsma.shape
    jsma_adv = metrics.accuracy_score(ytest_jsma, y_pred2)
    black_adv = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    print "Accuracy Score with LR is: ", "fgsm: ", fgsm_test, fgsm_adv, ", jsma: ", jsma_test, jsma_adv, ", black: ", black_test, black_adv 

    #################
    # Test with SVM
    classifier = OneVsRestClassifier(svm.SVC(gamma=0.001), n_jobs = 10)
    classifier.fit(reshape(x_train), reshape_y(y_train))

    y_pred1 = classifier.predict(reshape(x_test1))
    y_pred2 = classifier.predict(reshape(x_test2))
    y_pred3 = classifier.predict(reshape(x_test3))
    fgsm_test = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    jsma_test = metrics.accuracy_score(reshape_y(y_test2), y_pred2)
    black_test = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    y_pred1 = classifier.predict(reshape(x_adv1))
    y_pred2 = classifier.predict(reshape(x_adv2))
    y_pred3 = classifier.predict(reshape(x_adv3))
    fgsm_adv = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    jsma_adv = metrics.accuracy_score(ytest_jsma, y_pred2)
    black_adv = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    print "Accuracy Score with SVM is: ", "fgsm: ", fgsm_test, fgsm_adv, ", jsma: ", jsma_test, jsma_adv, ", black: ", black_test, black_adv 
    ################
    # Test with kNN
    neigh = KNeighborsClassifier(n_neighbors=5,n_jobs=10)
    neigh.fit(reshape(x_train),reshape_y(y_train))

    y_pred1 = neigh.predict(reshape(x_test1))
    y_pred2 = neigh.predict(reshape(x_test2))
    y_pred3 = neigh.predict(reshape(x_test3))
    fgsm_test = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    jsma_test = metrics.accuracy_score(reshape_y(y_test2), y_pred2)
    black_test = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    y_pred1 = neigh.predict(reshape(x_adv1))
    y_pred2 = neigh.predict(reshape(x_adv2))
    y_pred3 = neigh.predict(reshape(x_adv3))
    fgsm_adv = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    jsma_adv = metrics.accuracy_score(ytest_jsma, y_pred2)
    black_adv = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    print "Accuracy Score with kNN is: ", "fgsm: ", fgsm_test, fgsm_adv, ", jsma: ", jsma_test, jsma_adv, ", black: ", black_test, black_adv 
    ################
    #################
    # Test with Neural Networks
    mlp = MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-4,solver='sgd', tol=1e-4, random_state=1,learning_rate_init=.1)
    mlp.fit(reshape(x_train), reshape_y(y_train))

    y_pred1 = mlp.predict(reshape(x_test1))
    y_pred2 = mlp.predict(reshape(x_test2))
    y_pred3 = mlp.predict(reshape(x_test3))
    fgsm_test = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    jsma_test = metrics.accuracy_score(reshape_y(y_test2), y_pred2)
    black_test = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    y_pred1 = mlp.predict(reshape(x_adv1))
    y_pred2 = mlp.predict(reshape(x_adv2))
    y_pred3 = mlp.predict(reshape(x_adv3))
    fgsm_adv = metrics.accuracy_score(reshape_y(y_test1), y_pred1)
    jsma_adv = metrics.accuracy_score(ytest_jsma, y_pred2)
    black_adv = metrics.accuracy_score(reshape_y(y_test3), y_pred3)

    print "Accuracy Score with Neural Network is: ", "fgsm: ", fgsm_test, fgsm_adv, ", jsma: ", jsma_test, jsma_adv, ", black: ", black_test, black_adv 
    #################
    # Test with CNN
    '''
    tf.set_random_seed(5578)

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    sess = tf.Session()
    keras.backend.set_session(sess)

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model = baseline_model()
    predictions = model(x)

    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        #fgsm_test = model_eval(sess, x, y, predictions, x_test1, y_test1,args=eval_params)
        #fgsm_adv = model_eval(sess, x, y, predictions, x_adv1, y_test1,args=eval_params)


        print x_test2.shape, y_test2.shape, x_adv2.shape, ytest_jsma.shape

        jsma_test = model_eval(sess, x, y, predictions, x_test2, y_test2,args=eval_params)
        jsma_adv = model_eval(sess, x, y, predictions, x_adv2, get_yarray((len(ytest_jsma), 10),ytest_jsma),args=eval_params)
        #black_test = model_eval(sess, x, y, predictions, x_test3, y_test3,args=eval_params)
        #black_adv = model_eval(sess, x, y, predictions, x_adv3, y_test3,args=eval_params)
        #print "Accuracy Score with CNN is: ", "fgsm: ", fgsm_test, fgsm_adv, ", jsma: ", jsma_test, jsma_adv, ", black: ", black_test, black_adv
        print "Accuracy Score with CNN is: ", jsma_test, jsma_adv

    model_train(sess, x, y, predictions, x_train, y_train,evaluate=evaluate, args=train_params)



(x_train, y_train, x_test1, x_adv1, y_test1) = initialize("fgsm")    
(x_train, y_train, x_test2, x_adv2, y_test2) = initialize("jsma")    
(x_train, y_train, x_test3, x_adv3, y_test3) = initialize("black")
b = []
for a in x_adv2:
    b.append(a[0])
x_adv2 = np.array(b)


run(x_train, y_train, x_test1, x_adv1, y_test1, x_test2, x_adv2, y_test2, x_test3, x_adv3, y_test3)
