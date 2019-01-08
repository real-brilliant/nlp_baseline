# -*- coding: utf-8 -*-
# @auther: Geek_Fly
# @date  : 2018/12/28

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .basic_module import basic_module


class BiGruAtt(basic_module):
    def __init__(self, args, vectors):
        super(BiGruAtt, self).__init__()
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
        
        # att_score = softmax(W_proj * tanh(W * gru_output))
        self.weight_W = nn.Parameter(torch.Tensor(2*args.linear_hidden_size, 2*args.linear_hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(2*args.linear_hidden_size, 1))
        
        #self.fc = nn.Linear(2*args.linear_hidden_size, args.label_size)
        self.fc = nn.Sequential(
                nn.Linear(2*args.linear_hidden_size, 2*args.linear_hidden_size),
                nn.BatchNorm1d(2*args.linear_hidden_size),
                nn.ReLU(),
                nn.Linear(2*args.linear_hidden_size, args.label_size)
        )

        self.weight_W.data.uniform_(-0.1, 0.1)
        self.weight_proj.data.uniform_(-0.1, 0.1)


    def forward(self, input_ids, re):
        embeds = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        output, _ = self.bigru(embeds)      # [batch, seq_len, 2*hidden_size]
        #print(output[0])
        #print('shape of out = {}'.format(output.shape))
        #print('shape of out = {}'.format(output[0].shape))

        #squish = matmul_bias(output, self.weight_W,self.bias, nonlinearity='tanh')  # [batch, seq_len, 2*hidden_size]
        squish = torch.tanh(torch.matmul(output, self.weight_W))  # [batch, seq_len, embed_dim]
        #print('shape of squish = {}'.format(squish.shape))
        att = torch.matmul(squish, self.weight_proj)   # [batch, seq_len, 1]
        #print(att[1])
        #('shape of att = {}'.format(att.shape))  
        att_norm = F.softmax(att, dim=1)   # [batch, seq_len, 1]
        #print(att_norm[1])
        #print('shape of att_norm = {}'.format(att_norm.shape))   
        scored_output = output * att_norm    # [batch, seq_len, 2*hidden_size]
        #print(scored_output[0])
        #print('shape of scored_output = {}'.format(scored_output.shape))   
        feat = torch.sum(scored_output, dim=1)    # [batch, 2*hidden_size]
       #print('shape of feat = {}'.format(feat.shape)) 
        logit = self.fc(feat)

        return logit
