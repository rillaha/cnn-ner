# -*- coding: utf-8 -*-
"""
分词
"""
import sys
import jieba


def cut(input_file, output_file):
    # word_list = jieba.cut(text, cut_all=True)
    c_lines = ''
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # line = line.strip()
        c_line = ' '.join(jieba.cut(line, cut_all=False))
        c_lines += c_line.encode('utf-8')
    with open(output_file, 'w') as f:
        f.write(c_lines[:-1])


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    cut(input_path, output_path)
