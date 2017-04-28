import numpy as np
import scipy.misc
from matplotlib import pyplot as plt

X_test = np.load("xtest.npy")
X_test[5], X_test[7] = X_test[7], X_test[5]
X_test[6], X_test[8] = X_test[8], X_test[6]
X_test[7], X_test[11] = X_test[11], X_test[7]
X_test[8], X_test[30] = X_test[30], X_test[8]
X_test[9], X_test[61] = X_test[61], X_test[9]
np.save("xtest.npy",X_test)
for i in range(10):
    scipy.misc.imsave('jsma/exp'+str(i) + ".jpg", X_test[i].transpose(2,0,1)[0])
    #scipy.misc.imsave('black/adv' + str(i) + ".jpg", X2[i].transpose(2,0,1)[0])
