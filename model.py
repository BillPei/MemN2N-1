from __future__ import division
from __future__ import print_function

from reader import read_data

import os
import tensorflow as tf
import numpy as np


class QAModelN2N(object):
    def __init__(self, config, sess):
        self.nwords        = config.nwords
        self.mem_size      = config.mem_size
        self.vocab         = config.vocab
        self.edim          = config.edim
        self.nhop          = config.nhop
        self.init_std      = config.init_std
        self.sess          = sess
        self.init_lr       = config.init_lr
        self.max_grad_norm = config.max_grad_norm

        self.checkpoint_dir = config.checkpoint_dir
        self._attrs = ["edim", "mem_size", "nhop", "current_lr"]

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.inputs = tf.placeholder(tf.int32, [None], name="inputs")
        self.q      = tf.placeholder(tf.int32, [None], name="question")
        self.a      = tf.placeholder(tf.int32, [None], name="answer")
        self.lr     = tf.placeholder(tf.float32, name="learning_rate")

    def build_model(self):
        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.W = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.H = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

        # temporal encoding
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        self.T_C = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # partition input into sentences
        partitions = tf.scan(lambda acc, x: tf.select(tf.equal(x, 0), acc + 1, acc),
                             self.inputs, initializer=tf.constant(-1))
        part_data  = tf.dynamic_partition(self.inputs, partitions, self.mem_size)

        self.ms = []
        self.cs = []

        # TODO: put ps, and ms into tensors with pack to make use of more
        # efficient tensor operations (instead of lists)?
        # TODO: add position encoding
        # TODO: add random noise
        # TODO: add linear start
        # TODO: add batching
        for i,sent in enumerate(part_data):
            init = tf.zeros([self.edim])
            mi = tf.foldl(lambda acc, x: tf.nn.embedding_lookup(self.A, x) + acc,
                          sent, initializer=init)
            ci = tf.foldl(lambda acc, x: tf.nn.embedding_lookup(self.C, x) + acc,
                          sent, initializer=init)
            mi += tf.slice(self.T_A, [i, 0], [1, self.edim])
            ci += tf.slice(self.T_C, [i, 0], [1, self.edim])
            self.ms.append(mi)
            self.cs.append(ci)
        self.cs = tf.reshape(tf.pack(self.cs), [self.mem_size, self.edim])

        uk = tf.foldl(lambda acc, x: tf.nn.embedding_lookup(self.B, x) + acc,
                      self.q, initializer=tf.zeros([self.edim]))
        uk = tf.reshape(uk, [1, self.edim])
        for i in xrange(self.nhop):
            pk = []
            for mi in self.ms:
                prod = tf.matmul(uk, mi, transpose_b=True)
                pk.append(prod)
            pk = tf.nn.softmax(tf.reshape(tf.pack(pk), [1, len(pk)]))
            ok = tf.reduce_sum(tf.mul(tf.transpose(pk), self.cs), 0)
            uk = tf.nn.relu(tf.transpose(tf.matmul(self.H, uk, transpose_b=True)) + ok)

        self.logits = tf.transpose(tf.matmul(self.W, ok + uk, transpose_b=True))

        # for sampling
        self.a_hat = tf.nn.softmax(self.logits)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.a)

        # gradient stuff
        self.opt = tf.train.GradientDescentOptimizer(self.lr)
        #self.optim = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        # gradient clipping
        params = [self.A, self.B, self.C, self.W, self.H, self.T_A, self.T_C]
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1])
                                  for gv in grads_and_vars]
        self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.initialize_all_variables().run()
        tf.scalar_summary(["total loss"], self.loss)

    def sample(self, x, q, rev_vocab):
        a_hat = self.sess.run(self.a_hat, feed_dict={self.inputs: x, self.q: q})
        idx = np.argmax(a_hat)
        return rev_vocab[idx]

    def train(self, data_path, epochs=100):
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/{}".format(self.get_model_name()),
                                        self.sess.graph)

        data_size = 1
        for epoch in xrange(epochs):
            generator = read_data(data_path, self.vocab)
            lr = self.init_lr
            if epoch % 25 == 0 and 0 < epoch < 100:
                lr /= 2.
            for step,(x,q,a) in enumerate(generator):
                _, loss, summary = self.sess.run(
                    [self.optim, self.loss, merged_sum],
                    feed_dict={self.inputs: x, self.q: q, self.a: a, self.lr: lr})

                if step % 10 == 0:
                    print("Epoch: {}, Step: {}, loss: {}".format(epoch,
                          epoch*data_size + step, loss))
                if step % 2 == 0:
                    writer.add_summary(summary, epoch*data_size + step)
                if step % 500 == 0:
                    self.save(global_step=step)
            data_size = step + 1

    def get_model_name(self):
        model_name = "qa_model"
        for attr in self._attrs:
            if hasattr(self, attr):
                model_name += "-{}={}".format(attr, getattr(self, attr))
        return model_name

    def save(self, global_step=None):
        self.saver = tf.train.Saver()
        print(" [*] Saving checkpoints...")

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.get_model_name()),
                        global_step=global_step)

    def load(self, checkpoint_dir):
        self.saver = tf.train.Saver()

        print(" [*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" [*] Loaded model")
        else:
            print(" [!] Load failed")
