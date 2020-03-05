# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020-02-19 13:
"""

with open('./data/vocab.txt', 'r') as f:
    vl = [i for i in f.readlines()]
    vs = set(vl)

with open('./data/vocab.txt', 'w') as f:
    for i in vs:
        f.write(i)
