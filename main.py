# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2017-12-25
#
# Copyright (C) 2017 Taishi Matsumura
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

plt.close('all')

data = np.loadtxt('./data/experiment.csv', delimiter=',', skiprows=1)
eeg = data[:, 3:17]
eeg_ = eeg + np.array([[i*500 for i in xrange(14)] for j in xrange(len(eeg))])
marker = data[:, 24]


def plot_all_eeg(eeg):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(eeg)
    f.show()

fs = 2048
nyq = fs / 2.0
fe1 = 50 / nyq
fe2 = 500 / nyq
numtaps = 255
b = scipy.signal.firwin(numtaps, [fe1, fe2], pass_zero=False)
x = eeg[:, 0]
y = scipy.signal.lfilter(b, 1, x)

f = plt.figure()
ax1 = f.add_subplot(211)
ax2 = f.add_subplot(212)
ax1.plot(x)
ax2.plot(y)
f.show()
