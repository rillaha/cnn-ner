# -*- coding: utf-8 -*-
"""
Convert Traditional Chinese to Simplified Chinese
"""
import sys
import opencc


def t2s(input_path, output_path):
    s_lines = ''
    with open(input_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        s_line = opencc.convert(line)
        s_lines += s_line.encode('utf-8')
    with open(output_path, 'w') as f:
        f.write(s_lines[:-1])


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    t2s(input_path, output_path)
