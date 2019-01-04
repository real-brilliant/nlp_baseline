# -*- coding: utf-8 -*-
# @auther: Geek_Fly
# @date  : 2018/12/26

import os
import json
import jieba
import codecs
import pickle
import random
import argparse
import logging
from tqdm import tqdm, trange

import torch
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import models
from functions.build_emb import get_word_freq, get_embedding


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    '''A single training/test example for simple sequence classification.'''
    def __init__(self, guid, text, re, label=None):
        '''
        Args:
            guid : []
            text : []
            re   : []
            label: []
        '''
        self.guid = guid
        self.text = text
        self.re = re
        self.label = label


class InputFeatures(object):
    '''A single set of features of data.'''
    def __init__(self, input_ids, input_re, label_id):
        '''
        Args:
            input_ids: []
            input_re : []
            label_id : []
        '''
        self.input_ids = input_ids
        self.input_re = input_re
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_text_examples(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts


class MyPro(DataProcessor):
    '''my processor'''
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'val.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, 'test.json')), 'test')

    def get_labels(self):
        return[0, 1]

    def _create_examples(self, dicts, set_type):
        examples = []
        for idx, dict in enumerate(dicts):
            guid = "%s-%s" % (set_type, idx)
            text = dict['question']
            re = 0
            label = dict['label']
            examples.append(
                InputExample(guid=guid, text=text, re=re, label=label))
        return examples


def load_word_embedding(embed_path, 
                        token2id_cache='data/token2id_cache.pkl', 
                        embed_mat_cache='data/embed_mat_cache.pkl'):
    ''' 读取token2id和embed_mat

    Args:
        embed_path     : 词向量文件地址
        token2id_cache : token2id缓存地址
        embed_mat_cache: 嵌入矩阵缓存地址

    Returns:
        token2id : [dict]词转id的字典
        embed_mat: [ListOfList]嵌入矩阵
    '''
    if os.path.exists(token2id_cache) and os.path.exists(embed_mat_cache):
        print('*** load token2id and id2embed from cache ***')
        # 读cache
        with open(token2id_cache, 'rb') as inf:
            token2id = pickle.load(inf)
        with open(embed_mat_cache, 'rb') as inf:
            embed_mat = pickle.load(inf)
        
    else:
        print('*** generating token2id and id2embed from datasets ***')
        token_counter = get_word_freq('data/')
        token2id, embed_mat = get_embedding(
            embed_path, token_counter, freq_threshold=2, embed_dim=300)
        # 写cache
        with open(token2id_cache, 'wb') as outf:
            pickle.dump(token2id, outf)
        with open(embed_mat_cache, 'wb') as outf:
            pickle.dump(embed_mat, outf)

    return token2id, embed_mat


def convert_examples_to_features(examples, label_list, embed_path, max_seq_length, tokenizer=jieba, show_example=False):
    ''' 将输入文本转换成输入特征

    Args:
        examples      : [ListOfInputExample]输入文本，包含序号、文本、特征位、类别4个部分
        label_list    : [List]所有可能的类别
        embed_path    : [str]embedding文件地址
        max_seq_length: [int]最大句长
        tokenizer     : 分词器，目前只有jieba方式
        show_example  : 是否print展示数据

    Returns:
        features: [ListOfInputFeatures]输入特征，包含序号、token2id结果、映射后类别
    '''
    label_map = {}
    features = []
    re = 0

    token2id, _ = load_word_embedding(embed_path)

    for idx, label in enumerate(label_list):
        label_map[label] = idx

    for idx, example in enumerate(examples):
        tokens = jieba.lcut(example.text)
        ids = [token2id[token] if token in token2id.keys() else 0 for token in tokens]
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            ids = ids[:max_seq_length]
        else:
            while len(tokens) < max_seq_length:
                tokens.append('<pad>')
                ids.append(1)      

        label_id = label_map[example.label]
        if idx < 5 and show_example:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % ' '.join(tokens))
            logger.info("index: %s" % ' '.join([str(i) for i in ids]))
            logger.info("label: %s (id = %d)" % (str(example.label), label_id))

        features.append(
            InputFeatures(
                input_ids = ids,
                input_re = re,
                label_id = label_id))

    return features


def train(model, args, processor):
    ''' 模型训练
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    label_list = processor.get_labels()
    # 训练参数设置
    criterion = F.cross_entropy
    optimizer = model.get_optimizer(args.lr1, args.lr2, args.weight_decay)

    # 数据&特征导入
    train_examples = processor.get_train_examples(args.data_path)
    train_features = convert_examples_to_features(
        train_examples, label_list, args.embed_path, args.max_seq_length, show_example=True)

    num_train_steps = int(
        len(train_examples) / args.batch_size * args.epochs)
    logger.info("***** Running Training *****", )
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_re = torch.tensor([f.input_re for f in train_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    # torch.utils.data.TensorDataset(*tensors) - Dataset wrapping tensors.
    # Each sample will be retrieved by indexing tensors along the first dimension.
    train_data = TensorDataset(all_input_ids, all_re, all_label_id)
    # torch.utils.data.DataLoader(
    #     dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, 
    #     num_workers=0, collate_fn=<function default_collate>, pin_memory=False, 
    #     drop_last=False, timeout=0, worker_init_fn=None)
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    # 开始训练
    model.train()
    model_save_pth = args.model_save_dir + args.model + '.pth'
    best_score = 0
    flags = 0

    for _ in trange(int(args.epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            input_ids, re, label_id = batch

            optimizer.zero_grad()
            #print('shape of input_ids = {}'.format(input_ids.shape))
            #print('shape of label_id = {}'.format(label_id.shape))
            pred = model(input_ids, re)
            #print('shape of pred = {}'.format(pred.shape))
            #print('shape of prediction = {}'.format(pred.max(1)[1]))
            loss = criterion(pred, label_id)
            loss.backward()
            optimizer.step()
        
        # 保存最佳模型
        f1 = val(model, args)
        if f1 > best_score:
            best_score = f1
            print('* f1 score = {}'.format(f1))
            flags = 0
            checkpoint = {
                'state_dict': model.state_dict()
            }
            torch.save(checkpoint, model_save_pth)
        else:
            print('f1 score = {}'.format(f1))
            flags += 1
            if flags >= 6:
                break


def val(model, args):
    ''' validation '''   
    # 载入验证集数据
    processor = MyPro()
    label_list = processor.get_labels()
    val_examples = processor.get_dev_examples(args.data_path)
    val_features = convert_examples_to_features(
        val_examples, label_list, args.embed_path, args.max_seq_length)
    
    all_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
    all_re = torch.tensor([f.input_re for f in val_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
    val_data = TensorDataset(all_input_ids, all_re, all_label_id)
    # 生成iterator
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=3)

    # 开始验证
    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, re, label_id in val_dataloader:
        #print('shape of input_ids = {}'.format(input_ids.shape))
        #print('shape of label_id = {}'.format(label_id.shape))
        with torch.no_grad():
            logits = model(input_ids, re)
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_id.cpu().numpy()))

    f1 = np.mean(metrics.f1_score(predict, gt, average=None))

    return f1


def test(model, args):
    ''' test '''
    model_save_pth = args.model_save_dir + args.model + '.pth'
    model.load_state_dict(torch.load(model_save_pth)['state_dict'])
    
    # 载入验证集数据
    processor = MyPro()
    label_list = processor.get_labels()
    test_examples = processor.get_dev_examples(args.data_path)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.embed_path, args.max_seq_length, show_example=True)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_re = torch.tensor([f.input_re for f in test_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_re, all_label_id)
    # 生成iterator
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)

    # 开始验证
    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, re, label_id in test_dataloader:
        with torch.no_grad():
            logits = model(input_ids, re)
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_id.cpu().numpy()))

    f1 = np.mean(metrics.f1_score(predict, gt, average=None))

    return f1


def main():
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument("--task",
                        default = 'aDragon',
                        type = str,
                        help = "运行模式：aDragon/train/test")
    # 模型相关
    parser.add_argument("--model",
                        default = 'FastTextHiddenF',
                        type = str,
                        help = "所使用的模型名称")
    parser.add_argument("--model_save_dir",
                        default = 'checkpoints/',
                        type = str,
                        help = "模型保存目录")
    parser.add_argument("--embed_dim",
                        default = 300,
                        type = int,
                        help = "[自动获取]词向量维度")
    parser.add_argument("--vocab_size",
                        default = 0,
                        type = int,
                        help = "[自动获取]词表维度")
    parser.add_argument("--linear_hidden_size",
                        default = 50,
                        type = int,
                        help = "隐层维度")
    # 数据/特征相关参数
    parser.add_argument("--data_path",
                        default = 'data/',
                        type = str,
                        help = "数据所在目录")
    parser.add_argument("--embed_path",
                        default = 'data/acl_embeddings.txt',
                        type = str,
                        help = "embedding文件地址")
    parser.add_argument("--max_seq_length",
                        default = 15,
                        type = int,
                        help = "句子最大token数")
    parser.add_argument("--feature_dim",
                        default = 1,
                        type = int,
                        help = "特征位维度")
    parser.add_argument("--label_size",
                        default = 0,
                        type = int,
                        help = "[自动获取]类别个数")
    # 模型训练相关参数
    parser.add_argument("--lr1",
                        default = 1e-3,
                        type = float,
                        help = "lr1")
    parser.add_argument("--lr2",
                        default = 0.0,
                        type = float,
                        help = "lr2: embedding层学习率")
    parser.add_argument("--weight_decay",
                        default = 0.0,
                        type = float,
                        help = "weight_decay")
    parser.add_argument("--seed",
                        default = 777,
                        type = int,
                        help = "随机数种子")
    parser.add_argument("--batch_size",
                        default = 128,
                        type = int,
                        help = "每个batch的样本数量")
    parser.add_argument("--epochs",
                        default = 10,
                        type = int,
                        help = "最大epoch数")

    args = parser.parse_args()
    
    # 检查部分配置是否正确
    if args.task not in ['aDragon', 'train', 'test']:
        raise Exception('incorrect task name.', args.task)
    if args.model not in ['FastText', 'FastTextHiddenF']:
        raise Exception('incorrect model name.', args.model)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    
    # 配置[自动获取]参数
    # 根据processor修改label_size
    processor = MyPro()
    label_list = processor.get_labels()
    args.label_size = len(label_list)
    # 根据导入的embedding表修改vocab_size和embed_dim
    _, vectors = load_word_embedding(args.embed_path)
    args.vocab_size = len(vectors)
    args.embed_dim = len(vectors[0])    

    # 准备模型
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = getattr(models, args.model)(args, vectors)
    print(model)
    
    # 开始训练
    if args.task in ['aDragon', 'train']:
        train(model, args, processor)
    
    # 开始测试
    if args.task in ['aDragon', 'test']:    
        f1 = test(model, args)
        print('f1 score on test dataset is {}'.format(f1))
    

if __name__ == '__main__':
    main()
    
    '''
    processor = MyPro()
    label_list = processor.get_labels()
    print(label_list)

    train_examples = processor.get_train_examples('data/')
    print('*** get t_e success ***')
    convert_examples_to_features(
        train_examples, label_list, 15, tokenizer=jieba, show_example=True)
    '''
    