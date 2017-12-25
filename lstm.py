# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2017-12-15
#
# Copyright (C) 2017 Taishi Matsumura
#
import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import Chain
import chainer.functions
import chainer.links


def dataset(total_size, test_size):
    x, y = [], []
    for i in xrange(total_size):
        if xp.random.rand() <= 0.5:
            _x = xp.sin(xp.arange(xp.random.randint(10, 20)) + xp.random.rand())
            _x += xp.random.rand(len(_x)) * 0.05
            x.append(v(_x[:, xp.newaxis]))
            y.append(xp.array([1]))
        else:
            _x = xp.random.rand(xp.random.randint(10, 20))
            x.append(v(_x[:, xp.newaxis]))
            y.append(xp.array([0]))

    x_train =    x[:-test_size]
    y_train = vi(y[:-test_size])
    x_test =    x[-test_size:]
    y_test = vi(y[-test_size:])

    return x_train, x_test, y_train, y_test


def v(x):
    return chainer.Variable(xp.asarray(x, dtype=xp.float32))


def vi(x):
    return chainer.Variable(xp.asarray(x, dtype=xp.int32))


class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__()
        input_dim = 1
        hidden_dim = 5
        output_dim = 1

        with self.init_scope():
            self.lstm = chainer.links.NStepLSTM(
                    n_layers=1, in_size=input_dim,
                    out_size=hidden_dim, dropout=0.3)
            self.l1 = chainer.links.Linear(hidden_dim, hidden_dim)
            self.l2 = chainer.links.Linear(hidden_dim, output_dim)

    def __call__(self, xs):
        _, __, h = self.lstm(None, None, xs)
        h = v([_h[-1].data for _h in h])
        h = chainer.functions.relu(self.l1(h))
        y = self.l2(h)
        return chainer.functions.sigmoid(y)


def forward(x, y, model):
    t = model(x)
    loss = chainer.functions.sigmoid_cross_entropy(t, y)
    return loss


def loss_plot(train_loss, valid_loss):
    x = xp.arange(len(train_loss))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(x, train_loss)
    ax.plot(x, valid_loss)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    f.savefig('loss.png')


if __name__ == '__main__':
    gpu_device = 0
    chainer.cuda.get_device(gpu_device).use()
    xp = chainer.cuda.cupy

    max_epoch = 1000
    train_size = 1000
    valid_size = 1000

    model = RNN()
    model.to_gpu(gpu_device)

    x_train, x_test, y_train, y_test = dataset(
            train_size + valid_size, train_size)

    optimizer = chainer.optimizers.RMSprop(lr=0.03)
    optimizer.setup(model)

    early_stopping = 20
    min_valid_loss = 1e8
    min_epoch = 0

    train_loss, valid_loss = [], []

    for epoch in xrange(1, max_epoch):
        _y = model(x_test)
        y = _y.data
        y = xp.array([1 - y, y], dtype='f').T[0]
        accuracy = chainer.functions.accuracy(y, y_test.data.flatten()).data

        _train_loss = chainer.functions.sigmoid_cross_entropy(model(x_train), y_train).data
        _valid_loss = chainer.functions.sigmoid_cross_entropy(_y, y_test).data
        train_loss.append(_train_loss)
        valid_loss.append(_valid_loss)

        if min_valid_loss >= _valid_loss:
            min_valid_loss = _valid_loss
            min_epoch = epoch
        elif epoch - min_epoch >= early_stopping:
            break

        optimizer.update(forward, x_train, y_train, model)
        print('epoch: {}, acc: {}, loss: {}, valid_loss: {}'.format(epoch, accuracy, _train_loss, _valid_loss))

    loss_plot(train_loss, valid_loss)
    chainer.serializers.save_npz('model.npz', model)
