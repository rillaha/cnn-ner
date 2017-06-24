# -*- coding: utf-8 -*-
import os
import argparse
import log
import xmltools

TXT_DIR = 'xml2txts/'

logger = log.get_logger('w2c')

parser = argparse.ArgumentParser(
    description='文本准备',
    formatter_class=lambda prog: argparse.RawTextHelpFormatter(
        prog,
        max_help_position=50
    )
)

parser.add_argument(
    '-x',
    '--xml',
    help='xml文件地址',
)

parser.add_argument(
    '-o',
    '--out',
    help='输出文件地址',
)

SYS_ARGS = parser.parse_args()


def xml_to_txt():
    """将xml文件转换为txt文件"""
    total_count = 0
    sucess_count = 0
    for root, dirs, files in os.walk(SYS_ARGS.xml):
        for f in files:
            if f.find('.xml') == -1:
                continue
            total_count += 1
            xml_file = os.path.join(root, f)
            try:
                xmltools.xml2txt(xml_file, TXT_DIR, total_count)
                sucess_count += 1
            except Exception as e:
                logger.error(xml_file + ': ' + repr(e))
    logger.info(
        'end: xml2txt; sucess: {0}/{1}'.format(sucess_count, total_count))


def summary():
    """tx文件组合"""
    total_count = 0
    sucess_count = 0
    art = ''
    for root, dirs, files in os.walk(TXT_DIR):
        for f in files:
            if f.find('.txt') == -1:
                continue
            total_count += 1
            txt_file = os.path.join(root, f)
            try:
                with open(txt_file, 'r') as txt_f:
                    lines = txt_f.readlines()
                for line in lines:
                    if line.strip() == '':
                        continue
                    art += line
                art += '\n'
                sucess_count += 1
            except Exception as e:
                logger.error(txt_file + ': ' + repr(e))
    with open(SYS_ARGS.out, 'w') as f:
        f.write(art)
    # os.system('rm -rf ' + TXT_DIR)
    logger.info(
        'end: summary; sucess: {0}/{1}'.format(sucess_count, total_count))


if __name__ == '__main__':
    xml_to_txt()
    summary()
