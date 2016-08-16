from __future__ import division
from __future__ import print_function


from model import QAModelN2N
from reader import preprocess_data
from reader import read_data

import tensorflow as tf
import cPickle as pkl
import os


flags = tf.app.flags

flags.DEFINE_integer("edim", 150, "Embedding dimension [150]")
flags.DEFINE_integer("nhop", 6, "Number of hops [6]")
flags.DEFINE_integer("epochs", 100, "Number of epochs [100]")
flags.DEFINE_float("init_std", 0.05, "Initial standard deviation [0.05]")
flags.DEFINE_float("init_lr", 0.01, "Initial learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 50, "Clip gradients to norm [50]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Where to save checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "tasks_1-20_v1-2/en", "The data path [tasks_1-20_v1-2/en]")
flags.DEFINE_string("data_name", "qa1_single-supporting-fact_", "The name of the data [qa1_single-supporting-fact_]")

FLAGS = flags.FLAGS


def sample(FLAGS, n=10):
    data_path = os.path.join(FLAGS.data_dir, FLAGS.data_name + "test.txt")

    with open(os.path.join(FLAGS.checkpoint_dir, FLAGS.data_name + ".config"), "rb") as f:
        config = pkl.load(f)

    FLAGS.mem_size = config["mem_size"]
    FLAGS.nwords   = config["nwords"]
    FLAGS.vocab    = config["vocab"]

    with tf.Session() as sess:
        m = QAModelN2N(FLAGS, sess)
        m.build_model()
        m.load(FLAGS.checkpoint_dir)
        generator = read_data(data_path, m.vocab)
        rev_vocab = {v:k for k,v in m.vocab.iteritems()}
        for i,(x,q,a) in enumerate(generator):
            if i == n: break
            print("CONTEXT: " + " ".join([rev_vocab[xi] for xi in x]))
            print("QUESTION: " + " ".join([rev_vocab[xi] for xi in q]))
            print("PREDICTED ANSWER: " + m.sample(x, q, rev_vocab))
            print("ACTUAL ANSWER: " + rev_vocab[a[0]])
            print("="*80)


def main(_):
    data_path = os.path.join(FLAGS.data_dir, FLAGS.data_name + "train.txt")
    _, vocab, mem_size = preprocess_data(data_path)

    # storing data parameters
    # TODO: make reader object to save instead
    with open(os.path.join(FLAGS.checkpoint_dir, FLAGS.data_name + ".config"), "wb") as f:
        pkl.dump({"mem_size": mem_size, "nwords": len(vocab), "vocab": vocab}, f)

    FLAGS.mem_size = mem_size
    FLAGS.nwords = len(vocab)
    FLAGS.vocab = vocab

    with tf.Session() as sess:
        m = QAModelN2N(FLAGS, sess)
        m.build_model()
        m.train(data_path, epochs=FLAGS.epochs)


if __name__ == "__main__":
    #tf.app.run()
    sample(FLAGS)
