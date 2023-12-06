#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/9/26
'''
#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/9/26
'''
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Embedding(2, 2, input_length=7))
model.compile('rmsprop', 'mse')
p = model.predict(np.array([[0,1,0,1,1,0,0]]))
print(p)
print('--------------')
print(model.layers[0].W.get_value())