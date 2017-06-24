# -*- coding: utf8 -*-
"""
    通用工具
"""
import re
import xml.etree.cElementTree as et
from fetures import FEATURES
from fetures import YEARS


def get_filename(fill_name):
    """读取文件名，去除文件夹路径和后缀"""
    return '.'.join(fill_name.split('/')[-1].split('.')[:-1])


def get_fullname(fill_name):
    """获取文件名，将文件夹路径转化为前缀，去除后缀名"""
    return '.'.join(fill_name.replace('/', '_').split('.')[:-1])


def get_ext(file_name):
    """获取后缀名"""
    return file_name.split('.')[-1]


def get_feature(text):
    """获取特征向量"""
    feature_list = []
    for i in range(len(FEATURES)):
        if not text.find(FEATURES[i]) == -1:
            feature_list.append(1)
        else:
            feature_list.append(0)
    level = 0
    for y in YEARS:
        if not text.find(y) == -1:
            level += 1
    if level < 5:
        feature_list.append(0.2 * level)
    else:
        feature_list.append(1)
    return feature_list


def get_feature_from_xml(xml_file):
    """读取xml文件，获得特征向量"""
    xml_tree = et.ElementTree(file=xml_file)
    text = ''
    for child in xml_tree.getroot():
        if child.tag == 'resume' and child.text is not None:
            text = child.text.encode('utf-8').replace('\n', '')
    return get_feature(text)


def get_feature_from_txt(txt_file):
    """读取txt文件，获得特征向量"""
    with open(txt_file, 'r') as f:
        text = f.read().replace('\n', '')
    return get_feature(text)


def remove_punc(line):
    return re.sub('[：<>\-\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）\:]+'.decode('utf8'), ' ', line)
