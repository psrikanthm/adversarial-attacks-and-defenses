"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 3
men_means = (0.9048, 0.9027, 0.9034)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='g')

women_means = (0.1604, 0.7272, 0.8630)
rects2 = ax.bar(ind + width, women_means, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Convolutional Neural Network')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('FGSM', 'JBSM', 'Black Box'))

ax.legend((rects1[0], rects2[0]), ('Legitimate Examples', 'Adversarial Examples'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.show()
