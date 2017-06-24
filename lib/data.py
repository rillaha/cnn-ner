# -*- coding: utf -*-
import numpy as np
import jieba.posseg as pseg
from tensorflow.contrib import learn


def load_datasets(data_list):
    datasets = dict()
    datasets['data'] = []
    datasets['label_value'] = []
    datasets['label_name'] = ['others', 'bu', 'bf', 'mu', 'mf', 'du', 'df', 'title', 'hometown']
    for i in range(len(data_list)):
        text = list(open(data_list[i], 'r').readlines())
        data = [t.strip() for t in text]
        datasets['data'] += data
        datasets['target'] += [i for x in data]
    return datasets


def load_labels(datasets):
    text = datasets['data']
    lines = []
    for line in text:
        words = pseg.cut(line)
        words = [x.word.encode('utf-8') for x in words]
        lines.append(' '.join(words))
    labels = []
    for i in range(len(lines)):
        label = [0 for j in datasets['label_name']]
        label[datasets['label_value'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    max_document_length = max([len(x.split(' ')) for x in line])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(lines)))
    return [x, y, vocab_processor]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
