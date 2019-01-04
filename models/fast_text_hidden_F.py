# -*- coding: utf-8 -*-
# @auther: Geek_Fly
# @date  : 2019/01/02

import torch
from torch import nn
from .basic_module import basic_module

class FastTextHiddenF(basic_module):
    def __init__(self, args, vectors):
        super(FastTextHiddenF, self).__init__()
        self.args = args

        # （随机）初始化词嵌入矩阵：nn.Embedding(嵌入字典的大小，每个嵌入向量的大小)
        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)
        # 读取词嵌入矩阵vectors到预设好的embedding矩阵中
        self.embedding.weight.data.copy_(torch.tensor(vectors))

        self.pre = nn.Sequential(
                # nn.Linear(InputSize, OutputSize)
                nn.Linear(args.embed_dim, 2*args.embed_dim),
                nn.BatchNorm1d(2*args.embed_dim),
                nn.ReLU()
        )
        
        # ***************************************************************************
        # 注意(2*args.embed_dim + args.feature_dim)的args.feature_dim，是特征位re的长度
        # ***************************************************************************
        self.fc = nn.Sequential(
                nn.Linear(2*args.embed_dim + args.feature_dim, args.linear_hidden_size),
                nn.BatchNorm1d(args.linear_hidden_size),
                nn.ReLU(),
                nn.Linear(args.linear_hidden_size, args.label_size)
        )


    def forward(self, input_ids, re):
        #print('shape of input_ids = {}'.format(input_ids.shape))
        embed = self.embedding(input_ids)  # batch * seq * emb
        #print('shape of embed = {}'.format(embed.shape))
        embed_size = embed.size()
        # 调用.view前最好调用.contiguous()因为**view需要tensor的内存是整块d**
        # .view用于不改变数据的情况下改变矩阵形状，其实就是reshape? 不会改变原数据结构
        out = self.pre(embed.contiguous().view(-1, self.args.embed_dim)).view(embed_size[0], embed_size[1], -1)
        #print('shape of out = {}'.format(out.shape))
        mean_out = torch.mean(out, dim=1).squeeze()
        #print('shape of mean_out = {}'.format(mean_out.shape))        
 
        fc_input = torch.cat((
            mean_out.contiguous().view(-1, 2*self.args.embed_dim), 
            re.contiguous().view(-1, self.args.feature_dim).float()
            ), dim = 1)
        #print('shape of fc_input = {}'.format(fc_input.shape))
        logit = self.fc(fc_input)

        return logit

