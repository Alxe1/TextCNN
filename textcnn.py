#-*- coding: utf-8 -*-
# @Author  : LiuLei
# @File    : textcnn.py
# Desc     :
import datetime
import os
import pickle
import time
import jieba
import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

import data_helpers

logger = data_helpers.PrintLog("DeepTextCNN.log")

# with open("stopwords.txt", "r", encoding="utf-8") as f:
#     STOPWORDS = []
#     for sw in f:
#         STOPWORDS.append(sw.strip())
with open("stopwords_trie.txt", 'rb') as f:
    STOPWORDS = pickle.load(f)


class DeepTextCNN(object):

    def __init__(self, sequence_length, num_class, vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lamba=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(initial_value=tf.random_uniform(shape=[vocab_size, embedding_size],
                                                                 minval=-1.0, maxval=1.0,
                                                                 name="W"))
            self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedding_chars_expanded = tf.expand_dims(input=self.embedding_chars, axis=-1)

        # Convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

                conv = tf.nn.conv2d(  # TODO：可调整步长与padding
                    self.embedding_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                # Apply nonlinearty 'relu'
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # maxpooling over the output
                pooled = tf.nn.max_pool(  # TODO: 可调整padding和步长
                    value=h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name='pool'
                )
                pooled_outputs.append(pooled)
        # combine all the pooled feature
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob) # [None,num_filters_total]

        # final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                name="W",
                shape=[num_filters_total, num_class],
                initializer=tf.contrib.layers.xavier_initializer()  # tf.truncated_normal_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name='b')

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')  # shape: [batch_size, num_class]
            self.predictions = tf.argmax(input=self.scores, axis=1, name="predictions")  #shape: [batch_size]

        # calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lamba * l2_loss 

        # calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


def preprocess():
    '''
    Data Preparation
    :return:
    '''
    if not os.path.exists("seged_text.txt"):
        logger.info("需要重新生成数据seged_text.txt")

    # Load data
    logger.info("loading data...")
    x_text, y = data_helpers.load_seged_data(text_name="seged_text.txt", label_name='labels.txt')
    print(len(x_text), len(y))
    logger.info("loading data done...")

    # Split train/test dataset
    train_size = int(len(y) * 0.9)
    shuffled_y = np.random.permutation(range(len(y)))
    cnt = 0
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    for i in shuffled_y:
        if cnt <= train_size:
            train_set.append(x_text[i])
            train_labels.append(y[i])
            cnt += 1
        else:
            test_set.append(x_text[i])
            test_labels.append(y[i])
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    del shuffled_y

    with open("train_set_list.txt", 'wb') as f:
        pickle.dump(train_set, f)
    with open("train_labels.txt", 'wb') as f:
        pickle.dump(train_labels, f)
    with open("test_set_list.txt", 'wb') as f:
        pickle.dump(test_set, f)
    with open("test_labels.txt", 'wb') as f:
        pickle.dump(test_labels, f)

    # build vocabulary
    max_document_length = max([len(x.split(" ")) for x in train_set])
    if max_document_length >= 5120:
        max_document_length = 5120
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency=1)
    x = np.array(list(vocab_processor.fit_transform(train_set)))  # vocab_processor需要保存以备后用
    # print(x.shape)
    # random shuffle
    np.random.seed(10)
    logger.info("shuffle data...")
    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = train_labels[shuffle_indices]
    logger.info("shuffle data done...")

    # Split train/dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, train_size=0.8, random_state=1)

    del x, y, x_shuffled, y_shuffled  # 释放内存

    logger.info("Vocabulary size: {:d}".format(len(vocab_processor.vocabulary_)))
    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # save processed data
    with open("x_train.txt", 'wb') as f:
        pickle.dump(x_train, f)
    with open("y_train.txt", 'wb') as f:
        pickle.dump(y_train, f)
    with open("x_dev.txt", 'wb') as f:
        pickle.dump(x_dev, f)
    with open("y_dev.txt", 'wb') as f:
        pickle.dump(y_dev, f)

    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev=None, y_dev=None, learning_rate=1e-2):  # x_dev和y_dev暂时未用
    '''

    :param x_train:
    :param y_train:
    :param vocab_processor:
    :param x_dev:
    :param y_dev:
    :return:
    '''
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = DeepTextCNN(  # TODO: 调整参数
                sequence_length=x_train.shape[1],
                num_class=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=128,
                filter_sizes=[2,3,4,5],
                num_filters=128,
                l2_reg_lamba=0.0
            )
            # define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            ##---------------
            learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=200, decay_rate=0.95, staircase=True)  # 增加
            ##---------------
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss=cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output director for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.join("runs", timestamp)  # 修改路径，原os.path.abspath(os.path.join("runs", timestamp))
            logger.info("Writing to {}/n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summaries_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            ################### Dev summaries(can be added) #############################
            # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            # dev_summary_dir = os.path.join(out_dir, "summaries", 'dev')
            # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            #############################################################################

            # checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.join(out_dir, "checkpoints")  # 修改， 原os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)  # 保存模型个数，可以改变数量

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, keep_prob):
                """
                train step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict=feed_dict
                )
                # time_str = datetime.datetime.now().isoformat()
                logger.info("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
                train_summaries_writer.add_summary(summary=summaries, global_step=step)

            ####################### can be added dev_set here ########################
            # def dev_step(x_batch, y_batch, writer=None):
            #     feed_dict = {
            #         cnn.input_x: x_batch,
            #         cnn.input_y: y_batch,
            #         cnn.dropout_keep_prob: 1.0
            #     }
            #     step, summaries, loss, accuracy = sess.run(
            #         [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
            #         feed_dict=feed_dict
            #     )
            #     time_str = datetime.datetime.now().isoformat()
            #     print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #     if writer:
            #         writer.add_summary(summaries, step)
            ##########################################################################
            # Generate batches
            batches = data_helpers.batch_iter(  # TODO：调整参数
                list(zip(x_train,y_train)),
                batch_size=128,
                num_epochs=15,
                shuffle=True
            )
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch=x_batch, y_batch=y_batch, keep_prob=0.5)  # TODO: 调整参数keep_prob
                current_step = tf.train.global_step(sess, global_step)
                ################### can be added with dev_set ###################
                # if current_step % 20 == 0:
                #     print("\nEvaluation dev_set:")
                #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
                #################################################################
                if current_step % 100 == 0: 
                    path = saver.save(sess=sess,save_path=checkpoint_prefix, global_step=current_step)
                    logger.info("Save model checkpoint to {}\n".format(path))


def predict(x_test, y_test=None, checkpoint_dir=None):

    # Map data into vocabulary
    vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_test)))

    logger.info("\nEvaluating...\n")

    # Evaluation
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    logger.info(checkpoint_file)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensor we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), 64, 1, shuffle=False)  # tune the parameter

            # Collect the predictions
            all_predictions = np.array([], dtype=np.int64)

            for x_test_batch in batches:
                batch_prediction = sess.run(
                    predictions,
                    feed_dict={input_x: x_test_batch, dropout_keep_prob: 1.0}
                )
                all_predictions = np.concatenate([all_predictions, batch_prediction])
    print("all predictions: ", all_predictions)

    if y_test is not None:

        correct_predictions = np.sum(all_predictions == np.argmax(y_test, axis=1))
        accuracy = correct_predictions / len(y_test)
        print("test data accuracy: ", accuracy)
        print("reals: ", y_test)
        print("predictions: ", all_predictions)

    return all_predictions


def main():
    # logger.info("开始数据预处理")
    # start = time.time()
    # x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    # logger.info("数据预处理完成，用时：", str(time.time()-start))
    # logger.info("开始训练数据")
    # start = time.time()
    # train(x_train, y_train, vocab_processor, x_dev, y_dev)
    # logger.info("数据训练完成，用时：", str(time.time() - start))

    with open('test_set_list.txt', 'rb') as f:
        x_test = pickle.load(f)
    with open('test_labels.txt', 'rb') as f:
        y_test = pickle.load(f)
    print("数据长度：", (len(x_test), len(y_test)))
    print("预测:\n")
    predict(x_test, y_test, checkpoint_dir="runs\\1552289077\\checkpoints")


if __name__ == "__main__":
    main()
    


