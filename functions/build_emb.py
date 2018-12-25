# -*- coding: utf-8 -*-
# @auther: Geek_Fly
# @date  : 2018/12/25

import os
import json
import codecs
import jieba

from collections import Counter


def get_word_freq(file_path):
    ''' 统计文件出现的词频 
    
    Args:
        file_path: train、val、test文件所在目录

    Returns:
        token_counter: [dict]分词后词频统计结果
    '''
    token_counter = Counter()

    for file_name in ['train.json', 'val.json', 'test.json']:
        path = os.path.join(file_path, file_name)

        with codecs.open(path, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = json.loads(inf.strip())

                for token in jieba.lcut(inf['question']):
                    token_counter[token] += 1

    print("*** {} words in total ***".format(len(token_counter)))

    return token_counter


def get_embedding(file_path, token_counter, freq_threshold, embed_path, embed_dim):
    ''' 读取词向量 

    Args:
        file_path     : embedding文件所在目录
        token_counter : [dict]分词后词频统计结果
        freq_threshold: [int]词频最低阈值，低于此阈值的词不会进行词向量抽取
        embed_path    : embedding文件名
        embed_dim     : [int]词向量维度

    Returns:
        token2id : [dict]词转id的字典
        embed_mat: [ListOfList]嵌入矩阵
    '''
    embed_dict = {}
    filtered_elements = [k for k, v in token_counter.items() if v >= freq_threshold]
    
    path = os.path.join(file_path, embed_path)
    with codecs.open(path, 'r', 'utf-8') as infs:
        for inf in infs:
            inf = inf.strip()
            inf_list = inf.split()
            token = ''.join(inf_list[0:-embed_dim])

            if token in token_counter and token_counter[token] >= freq_threshold:
                embed_dict[token] = list(map(float, inf_list[-embed_dim:]))

    print("{} / {} tokens have corresponding embedding vector".format(
        len(embed_dict), len(filtered_elements)))

    unk = "<unk>"
    pad = "<pad>"
    # enumerate(iterable, start=0)，start代指起始idx（不影响token输出）
    token2id = {token: idx for idx, token in enumerate(embed_dict.keys(), 2)}
    token2id[unk] = 0
    token2id[pad] = 1
    embed_dict[unk] = [0. for _ in range(embed_dim)]
    embed_dict[pad] = [0. for _ in range(embed_dim)]

    id2embed = {id: embed_dict[token] for token, idx in token2id.items()}

    embed_mat = [id2embed[idx] for idx in range(len(id2embed))]

    return token2id, embed_mat

