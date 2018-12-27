# -*- coding: utf-8 -*-
# @auther: Geek_Fly
# @date  : 2018/12/27

import torch

class basic_module(torch.nn.Module):
    def __init__(self):
        super(basic_module, self).__init__()
        self.model_name = str(type(self))


    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        embed_params = list(map(id, self.embedding.parameters()))
        base_params = filter(lambda p: id(p) not in embed_params, self.parameters())
        optimizer = torch.optim.Adam([
            {'params':self.embedding.parameters(), 'lr':lr2},
            {'params':base_params, 'lr':lr1, 'weight_decay':weight_decay}
        ])
        return optimizer
