#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/9/26
'''

import torch
from torch import nn
from torch.autograd import Variable
# 定义词嵌入
embeds = nn.Embedding(2, 5) # 2 个单词，维度 5
# 得到词嵌入矩阵,开始是随机初始化的
torch.manual_seed(1)
print(embeds.weight)