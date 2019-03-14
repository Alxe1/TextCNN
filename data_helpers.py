#-*- coding: utf-8 -*-
# @Author  : LiuLei
# @File    : data_helpers.py
# @Software: PyCharm
# @Time    : 2019/2/28 14:10
# Desc     :
import logging
import time

import jieba
import os
# import marisa_trie
import pickle
import numpy as np
import pandas as pd
import gensim
from gensim.models import word2vec
from tensorflow.contrib import learn


def get_rawdata(sql, engine):
    '''

    :param sql:
    :param engine:
    :return:
    '''
    raw_df = pd.read_sql(sql, con=engine)
    idf = raw_df[['content', 'emotion']].itertuples(index=False, name=False)
    return idf


def cut_words(idf):
    '''

    :param idf:
    :return:
    '''
    labels = []

    if not os.path.exists("stopwords_trie.txt"):
        stopwords = get_stopwords(stop_words="stopwords.txt", serialize=True)
    else:
        with open("stopwords_trie.txt", 'rb') as f:
            stopwords = pickle.load(f)

    with open("seged_text_without_sw.txt", 'w', encoding='utf-8') as f:
        for content, label in idf:
            words = jieba.cut(content, cut_all=False, HMM=True)
            text = ""
            for w in words:
                if w not in stopwords:
                    text = " ".join([text, w])
            text += "\n"
            f.write(text)
            labels.append(label)

    with open("labels.txt", 'wb') as f:
        pickle.dump(labels, f)


def get_stopwords(stop_words="stopwords.txt", serialize=True):
    '''

    :param stop_words:
    :param serialize:
    :return:
    '''
    stopwords = []
    try:
        with open(stop_words, "r", encoding="utf-8") as f:
            for sw in f:
                stopwords.append(sw.strip())
        stopwords = marisa_trie.Trie(stopwords)
    except Exception as e:
        print(e, "加载停止词列表失败")

    if serialize:
        try:
            with open("stopwords_trie.txt", 'wb') as f:
                pickle.dump(stopwords, f)
        except Exception as e:
            print(e, "保存字典树失败")
    return stopwords


def load_seged_data(text_name="seged_text_without_sw.txt", label_name='labels.txt'):
    '''

    :param text_name:
    :param label_name:
    :return:
    '''
    with open(text_name, 'r', encoding='utf-8') as f:
        X = [line.strip() for line in f.readlines()]

    with open(label_name, 'rb') as f:
        labels = pickle.load(f)

    y_len = len(labels)
    y = np.zeros((y_len, 3))
    for i in range(y_len):
        if labels[i] == 1:
            y[i, 0] = 1
        elif labels[i] == -1:
            y[i, 1] = 1
        else:
            y[i, 2] = 1

    return [X, y]


def batch_iter(datas, batch_size, num_epochs, shuffle=True):
    '''

    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    '''
    data = np.array(datas)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    cnt = 1
    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        print("epoch %s" % cnt)
        cnt += 1
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def data_segment(rawdata, segedtext_path, segdlabel_path):
    '''

    :param rawdata:
    :param segedtext_path:
    :param segdlabel_path:
    :return:
    '''
    labels = []
    if not os.path.exists("stopwords_trie.txt"):
        stopwords = get_stopwords(stop_words="stopwords.txt", serialize=True)
    else:
        with open("stopwords_trie.txt", 'rb') as f:
            stopwords = pickle.load(f)

    with open(rawdata, 'r', encoding='utf-8') as rf:
        with open(segedtext_path, 'w', encoding='utf-8') as wf:
            for i, line in enumerate(rf.readlines()):
                split_line = line.split("##TAP##")
                # print(split_line)
                try:
                    label = int(split_line[0])
                except ValueError:
                    print("文本第一列不是label，排除该行 %s，进行下一行" % (i+1))
                    continue
                except Exception as e:
                    print("wrong: ",e)
                    continue
                try:
                    raw_text = "".join([split_line[1],split_line[2]]).strip()
                    seged_words = jieba.cut(raw_text)
                    seged_text = ""
                    for w in seged_words:
                        if w not in stopwords and w != " ":
                            seged_text = " ".join([seged_text, w])
                    seged_text += "\n"
                except  Exception as e:
                    print("第 %s 出错" % (i+1), e)
                    continue
                try:
                    wf.write(seged_text)
                    labels.append(label)
                except Exception as e:
                    print(e)

    with open(segdlabel_path, 'wb') as f:
        pickle.dump(labels, f)

def corpus_segment(corpus, segedcorpus_path):
    '''

    :param corpus:
    :param segedcorpus_path:
    :return:
    '''
    with open(corpus, 'r', encoding='utf-8') as rf:
        with open(segedcorpus_path, 'w', encoding='utf-8') as wf:
            for i, line in enumerate(rf.readlines()):
                split_line = line.split("##B##")
                # print(split_line)
                try:
                    if len(split_line) == 2:
                        raw_text = "".join([split_line[0], split_line[1]]).strip()
                        seged_words = jieba.cut(raw_text)
                        seged_text = " ".join(seged_words) + " \n"
                        wf.write(seged_text)
                except Exception as e:
                    print("第%s行出错" % (i+1), e)
                    continue


def PrintLog(log_name):
    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # log_name = "deeptextcnn.log"

    file_log = logging.FileHandler(log_name, mode='w')
    file_log.setLevel(level=logging.INFO)
    console_log = logging.StreamHandler()
    console_log.setLevel(level=logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    file_log.setFormatter(formatter)
    console_log.setFormatter(formatter)
    logger.addHandler(file_log)
    logger.addHandler(console_log)

    return logger


if __name__ == "__main__":
    X, y = load_seged_data("seged_text.txt", "labels.txt")

    # model = gensim.models.Word2Vec.load("SA_corpus/Hcorpus.model")
    # result = model.wv.most_similar("足球")
    # print(result)
    # print(model['足球'])

    # print('segmenting...')
    # t = time.time()
    # corpus_segment("SA_corpus/Hcorpus.txt", "SA_corpus/seged_Hcorpus.txt")
    # print("分词完成, ", time.time()-t)
    # corpus = word2vec.Text8Corpus("SA_corpus/seged_Hcorpus.txt")
    # print("training...")
    # t = time.time()
    # model = word2vec.Word2Vec(corpus, size=100, window=5, min_count=5)  # TODO:可以参数调优
    # print("训练完成, ", time.time() - t)
    # model.save("SA_corpus/Hcorpus.model")

    # max_document_length = 4
    # x_text = [
    #     'i love you',
    #     'me too'
    # ]
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # vocab_processor.fit(x_text)
    # print(next(vocab_processor.transform(['foo him'])).tolist())
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    # print(x)
    # print(len(vocab_processor.vocabulary_))

