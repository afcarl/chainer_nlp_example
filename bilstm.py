#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
random.seed(1234)
np.random.seed(1234)

import chainer
from chainer import Chain, cuda
from chainer import function, functions, links, optimizer
from chainer import Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import math
from chainer import initializers

import six

class FastBiLSTM(chainer.Chain):

    def __init__(self, n_vocab=None, emb_dim=100, hidden_dim=200,
                 init_emb=None, add_dim=0, use_dropout=0.33, n_layers=1,
                 pos_dim=0, n_pos=0):
        feature_dim = emb_dim + add_dim + pos_dim
        super(FastBiLSTM, self).__init__(
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            bi_lstm=L.NStepBiLSTM(n_layers=n_layers, in_size=feature_dim,
                                  out_size=hidden_dim, dropout=use_dropout,
                                  use_cudnn=True)
        )
        if n_pos:
            pos_embed = L.EmbedID(n_pos, pos_dim, ignore_label=-1)
            self.add_link('pos_embed', pos_embed)

        self.n_pos = n_pos
        self.hidden_dim = hidden_dim
        self.train = True
        self.use_dropout = use_dropout
        self.n_layers = n_layers

        # Forget gate bias => 1.0
        # MEMO: Values 1 and 5 reference the forget gate.
        for w in self.bi_lstm:
            w.b1.data[:] = 1.0
            w.b5.data[:] = 1.0

    def set_train(self, train):
        self.train = train

    def __call__(self, x_data, add_pos=None, add_h=None):

        batchsize = len(x_data)
        h_shape = (self.n_layers, batchsize, self.hidden_dim)
        hx = None
        cx = None

        xs_f = []
        for i, x in enumerate(x_data):
            # _x = self.xp.array(x, dtype=self.xp.int32)
            _x = x
            _x = Variable(_x, volatile=not self.train)
            _x = self.word_embed(_x)
            if self.n_pos:
                pos_vec = self.pos_embed(add_pos[i])
                _x = F.concat([_x, pos_vec], axis=1)
            _x = F.dropout(_x, ratio=self.use_dropout, train=self.train)
            x_f = _x
            xs_f.append(x_f)

        _hy_f, _cy_f, ys_f = self.bi_lstm(hx=hx, cx=cx, xs=xs_f,
                                          train=self.train)

        if add_h is not None:
            # BIO vectors
            ys = [F.dropout(F.concat([_f, add_h[i]]),
                            ratio=self.use_dropout, train=self.train)
                  for i, (_f) in enumerate(ys_f)]
        else:
            ys = [F.dropout(_f, ratio=self.use_dropout, train=self.train)
                  for i, _f in enumerate(ys_f)]

        return ys



if __name__ == '__main__':
    x_list = [[0, 1, 2, 3], [4, 5, 6], [7, 8]]
    x_list = [np.array(x, dtype=np.int32) for x in x_list]
    
    bilstm = FastBiLSTM(n_vocab=50)
    h_vecs = bilstm(x_list)
    print h_vecs
