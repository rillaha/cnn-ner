# -*- coding: utf-8 -*-
"""
Demo for word2vec
"""
import sys
import gensim
# model = gensim.models.Word2Vec.load('model/demo.model')
model = gensim.models.Word2Vec.load(sys.argv[1])


def similar():
    """ show similar words"""
    word = input('word: ')
    result = model.most_similar(word.decode('utf-8'))
    for e in result:
        print e[0], e[1]


def showVector():
    word = input('word: ')
    result = model[word.decode('utf-8')].shape()
    print result



if __name__ == '__main__':
    while True:
        similar()
    # showVector()
    # model.similarity(u"计算机", u"自动化")
    # print model.doesnt_match(u"早餐 晚餐 午餐 中心".split())
