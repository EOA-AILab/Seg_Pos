
#!/usr/bin/env output
# encoding: utf-8
'''
@Author: Joven Chu
@Email: jovenchu03@gmail.com
@File: pos_tag.py
@Time: 2019/5/21 11:39
@Project: bert-ner
@About:集成pkuseg、thulac、jieba、hanlp、pyltp的分词和词性标注
# pkuseg（1.53s）: 分词word_list: [('屠呦呦', 'nr'), ('，', 'w'), ('女', 'b'), ('，', 'w'), ('汉族', 'nz'), ('，', 'w'), ('中共', 'j'), ('党员', 'n'), ('，', 'w'), ('药学家', 'n'), ('。', 'w'), ('1930年12月30日', 't'), ('生于', 'v'), ('浙江宁波', 'ns'), ('，', 'w'), ('1951年', 't'), ('考入', 'v'), ('北京大学', 'nt'), ('，', 'w'), ('在', 'p'), ('医学院药学系', 'n'), ('生药', 'n'), ('专业', 'n'), ('学习', 'v'), ('。', 'w'), ('1955年', 't'), ('，', 'w'), ('毕业于', 'v'), ('北京医学院', 'n'), ('（', 'w'), ('今', 't'), ('北京大学', 'nt'), ('医学部', 'n'), ('）', 'w'), ('。', 'w')]
# thulac（1.39s）: 分词word_list: [['屠呦呦', 'uw'], ['，', 'w'], ['女', 'a'], ['，', 'w'], ['汉族', 'nz'], ['，', 'w'], ['中共', 'j'], ['党员', 'n'], ['，', 'w'], ['药学', 'n'], ['家', 'n'], ['。', 'w'], ['1930年12月30日', 'uw'], ['生于', 'v'], ['浙江宁波', 'uw'], ['，', 'w'], ['1951年', 't'], ['考入', 'v'], ['北京大学', 'ni'], ['，', 'w'], ['在', 'p'], ['医学院药学系', 'uw'], ['生药', 'n'], ['专业', 'n'], ['学习', 'v'], ['。', 'w'], ['1955年', 't'], ['，', 'w'], ['毕业', 'v'], ['于', 'p'], ['北京医学院', 'uw'], ['（', 'w'], ['今', 'g'], ['北京大学医学部', 'uw'], ['）', 'w'], ['。', 'w']]
# jieba（0.022s）: 分词word_list: [('屠呦呦', 'x'), ('，', 'x'), ('女', 'b'), ('，', 'x'), ('汉族', 'nz'), ('，', 'x'), ('中共党员', 'n'), ('，', 'x'), ('药学', 'n'), ('家', 'q'), ('。', 'x'), ('1930年12月30日', 'x'), ('生于', 'v'), ('浙江宁波', 'x'), ('，', 'x'), ('1951年', 'x'), ('考入', 'v'), ('北京大学', 'nt'), ('，', 'x'), ('在', 'p'), ('医学院药学系', 'x'), ('生药', 'n'), ('专业', 'n'), ('学习', 'v'), ('。', 'x'), ('1955年', 'x'), ('，', 'x'), ('毕业', 'n'), ('于', 'p'), ('北京医学院', 'x'), ('（', 'x'), ('今', 'zg'), ('北京大学医学部', 'nt'), ('）', 'x'), ('。', 'x')]
# pyltp（0.021s）: 分词word_list: [['屠呦呦', 'nh'], ['，', 'wp'], ['女', 'b'], ['，', 'wp'], ['汉族', 'nz'], ['，', 'wp'], ['中共', 'j'], ['党员', 'n'], ['，', 'wp'], ['药学家', 'n'], ['。', 'wp'], ['1930年', 'nt'], ['12月', 'nt'], ['30日', 'nt'], ['生于', 'v'], ['浙江宁波', 'ns'], ['，', 'wp'], ['1951', 'm'], ['年', 'q'], ['考入', 'v'], ['北京大学', 'ni'], ['，', 'wp'], ['在', 'p'], ['医学院', 'n'], ['药学系', 'n'], ['生药', 'n'], ['专业', 'n'], ['学习', 'v'], ['。', 'wp'], ['1955', 'm'], ['年', 'q'], ['，', 'wp'], ['毕业', 'v'], ['于', 'p'], ['北京医学院', 'ns'], ['（', 'wp'], ['今', 'nt'], ['北京大学', 'ni'], ['医学部', 'n'], ['）', 'wp'], ['。', 'wp']]

结论：1.一般使用jieba+命名实体作为自定义词典，便可以保证较高的准确率和产品速度。
     2.pkuseg和thulac的初始化模型就需要很长时间，而且自定义词典必须每次加载模型。不建议使用
     3.pyltp无法使用，且分词效果太细化。无法用到命名实体识别的自定义词典，因为LTP的分词模块并非采用词典匹配的策略，外部词典以特征方式加入机器学习算法，并不能保证所有的词都是按照词典里的方式进行切分。
'''


import os
from datetime import datetime
import pkuseg
import thulac
import jieba
import jieba.posseg as psg
from pyltp import Segmentor,Postagger

# pyltp语言模型加载
LTP_DATA_DIR = 'ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  #加载模型
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型


def pkuseg_pos(string):
    print('PkuSeg的分词和词性标注：')
    num = len(string)
    print(num)
    start_time = datetime.now()
    for s in string:
        seg = pkuseg.pkuseg(postag=True)  # 加载模型，给定用户词典
        pos_list = seg.cut(s)
    all_time = (datetime.now() - start_time).total_seconds()
    avg = all_time / num
    print('pos_tag time used: {} sec'.format(avg))
    print('\n\n')



def thulac_pos(string):
    print('THULAC的分词和词性标注：')
    num = len(string)
    print(num)
    start_time = datetime.now()
    for s in string:
        seg = thulac.thulac()  # 加载模型，给定用户词典
        pos_list = seg.cut(s)
    all_time = (datetime.now() - start_time).total_seconds()
    avg = all_time / num
    print('pos_tag time used: {} sec'.format(avg))
    print('\n\n')


def jieba_pos(string):
    print('Jieba的分词和词性标注：')
    num = len(string)
    print(num)
    start_time = datetime.now()
    for s in string:
        pos_list = ([(x.word,x.flag) for x in psg.lcut(s)])
    all_time = (datetime.now() - start_time).total_seconds()
    avg = all_time / num
    print('pos_tag time used: {} sec'.format(avg))
    print('\n\n')



def pyltp_pos(string):
    print('Pyltp的分词和词性标注：')
    num = len(string)
    print(num)
    start_time = datetime.now()
    for s in string:
        words = segmentor.segment(s)
        word_list = []
        for w in words:
            word_list.append(w)
        postags = postagger.postag(word_list)  # 词性标注
    all_time = (datetime.now() - start_time).total_seconds()
    avg = all_time / num
    print('pos_tag time used: {} sec'.format(avg))
    print('\n\n')



if __name__ == '__main__':
    string = []
    with open('person.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            string.append(line)
            
    # ************pkuseg（1.53s）*****************
    pkuseg_pos(string)

    # ************thulac（1.39s）********************
    thulac_pos(string)

    # *************jieba（0.022s）*****************
    jieba_pos(string)

    # ************pyltp（首次加载模型耗时0.573s，预测0.021s）********
    pyltp_pos(string)