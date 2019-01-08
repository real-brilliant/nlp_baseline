# -*- coding: utf-8 -*-
# @auther: Geek_Fly
# @date  : 2019/01/08

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .basic_module import basic_module


class BiGru(basic_module):
    def __init__(self, args, vectors):
        super(BiGru, self).__init__()
        self.args = args

        # （随机）初始化词嵌入矩阵：nn.Embedding(嵌入字典的大小，每个嵌入向量的大小)
        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)
        # 读取词嵌入矩阵vectors到预设好的embedding矩阵中
        self.embedding.weight.data.copy_(torch.tensor(vectors))

        self.bigru = nn.GRU(
            input_size = args.embed_dim,
            hidden_size = args.linear_hidden_size,
            bidirectional = True,
        )
        
        self.fc = nn.Sequential(
                nn.Linear(2*args.linear_hidden_size, 2*args.linear_hidden_size),
                nn.BatchNorm1d(2*args.linear_hidden_size),
                nn.ReLU(),
                nn.Linear(2*args.linear_hidden_size, args.label_size)
        )


    def forward(self, input_ids, re):
        embeds = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        output, _ = self.bigru(embeds)      # [batch, seq_len, 2*hidden_size]

        feat = torch.sum(output, dim=1)    # [batch, 2*hidden_size]
       #print('shape of feat = {}'.format(feat.shape)) 
        logit = self.fc(feat)

        return logit