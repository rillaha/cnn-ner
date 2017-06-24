# -*- coding: utf8 -*-
"""
    xml工具
"""
import os
import jieba
import xml.etree.cElementTree as et
from utils import remove_punc


def read_xml(xml_file):
    """读取xml 转换为字典"""
    xml_dict = {}
    xml_tree = et.ElementTree(file=xml_file)
    for child in xml_tree.getroot():
        if child.tag == 'name':
            xml_dict['name'] = xml_file[:-4]
            if child.text is not None:
                xml_dict['name'] = child.text.encode(
                    'utf-8').replace('\n', '').strip()
        if child.tag == 'org':
            xml_dict['org'] = ''
            if child.text is not None:
                xml_dict['org'] = child.text.encode(
                    'utf-8').replace('\n', '').strip()
        if child.tag == 'resume':
            xml_dict['resume'] = ''
            xml_dict['lines'] = []
            if child.text is not None:
                xml_dict['resume'] = child.text.encode(
                    'utf-8').replace('\n', '').strip()
                xml_dict['lines'] = child.text.encode('utf-8').split('\n')
    return xml_dict


def xml2txt(xml_file, output_dir, id):
    """转换xml为txt文件，创建空.ann文件(可选)"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    xml_dict = read_xml(xml_file)
    # ann文件
    if True:
        ann_file = '{0}/{1}.ann'.format(output_dir, id)
        print ann_file
        with open(ann_file, 'w') as f:
            f.write('')

    # 写txt文件
    txt_file = '{0}/{1}.txt'.format(output_dir, id)
    with open(txt_file, 'w') as f:
        f.write('')
    with open(txt_file, 'a+') as f:
        f.write('姓名 ' + xml_dict['name'] + ' 部门 ' + xml_dict['org'] + '\n')
        resume_length = len(xml_dict['lines'])
        for line_num in range(resume_length):
            line = xml_dict['lines'][line_num].strip()
            line = remove_punc(line.decode('utf-8')).encode('utf-8')
            if len(line) == 0:
                continue
            if not line_num == resume_length - 2:
                line += '\n'
            f.write(line)


def list2map(word_list, map_list=[]):
    """词汇列表 => 词频映射"""
    # 去重
    word_list = [word for word in set(word_list)]
    for word in word_list:
        word = word.encode('utf-8')
        if word.strip() == '' or word == ',':
            continue
        is_exist = False
        for map in map_list:
            if map['key'] == word:
                map['value'] += 1
                is_exist = True
                break
        if not is_exist:
            map_list.append({'key': word, 'value': 1})
    return sorted(map_list, key=lambda d: d['value'], reverse=True)


def output(map_list, output_path):
    """输出词频表"""
    with open(output_path, 'w') as fout:
        fout.write('')
    with open(output_path, 'a+') as fout:
        for word in map_list:
            fout.write('{0},{1}\n'.format(word['key'], word['value']))


def xml2map(xml_file, map_list):
    """统计词频"""
    xml_tree = et.ElementTree(file=xml_file)
    for child in xml_tree.getroot():
        if child.tag == 'resume':
            text = child.text.encode('utf-8').replace('\n', '')
    word_list = jieba.cut(text, cut_all=True)
    return list2map(word_list, map_list)
