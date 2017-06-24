#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import datetime
import tensorflow as tf
import numpy as np
import lib

DATA_LIST = [
    'data/others', 'data/bu', 'data/bf', 'data/mu', 'data/mf', 'data/du',
    'data/df', 'data/title', 'data/hometown'
]

# flags
FLAGS = lib.load_flags()
FLAGS._parse_flags()
print '{:=^37}'.format(' START ')
print '+------------------------+----------+'
print '| PARAMETER              | VALUE    |'
print '+------------------------+----------+'
for key, value in sorted(FLAGS.__flags.items()):
    print '| {:22} | {:<8} |'.format(key.upper(), value)
print '+------------------------+----------+'

# data
print '{:=^37}'.format(' LOAD DATA ')
print '+-----------------------------------+'
for data in DATA_LIST:
    print '| {:<33} |'.format(data)
print '+-----------------------------------+'
datasets = lib.load_datasets(DATA_LIST)
x, y, vocab_processor = lib.load_labels(datasets)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


def train_step(x_batch, y_batch):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss,
         cnn.accuracy], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print '{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss,
                                                    accuracy)
    train_summary_writer.add_summary(summaries, step)


def dev_step(x_batch, y_batch, writer=None):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print '{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss,
                                                    accuracy)
    if writer:
        writer.add_summary(summaries, step)


# training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = lib.CNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.word2vec_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    '{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    '{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, 'runs', timestamp))

        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        vocab_processor.save(os.path.join(out_dir, 'vocab'))

        sess.run(tf.global_variables_initializer())
        vocabulary = vocab_processor.vocabulary_
        initW = lib.load_embedding
        sess.run(cnn.W.assign(initW))

        batches = lib.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print '{:^37}'.format(' Evaluation ')
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(
                    sess, checkpoint_prefix, global_step=current_step)
