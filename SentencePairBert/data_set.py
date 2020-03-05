# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020-02-19 13:
"""
import random

with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_en.txt', 'r') as f:
    en = [i for i in f.readlines()]
    en1 = en[0:10197561]
    en2 = en[0:10197561]
    en3 = en[20197561:30197561]
with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_zh.txt', 'r') as f:
    zh = [i for i in f.readlines()]
    zh1 = zh[0:10197561]
    zh2 = zh[0:10197561]
    zh3 = zh[20197561:30197561]
label1 = ['1'] * 10197561
label2 = ['0'] * 10197561
label3 = ['0'] * 10000000
data = [(e, z, l) for e, z, l in zip(en1, zh1, label1)]
data = data + [(e, z, l) for e, z, l in zip(en2, zh2, label2)]
data = data + [(e, z, l) for e, z, l in zip(en3, zh3, label3)]
random.shuffle(data)
with open('/data/model/SentencePairMatch/tmpdata/en.train', 'w') as f:
    for i in [i[0] for i in data]:
        f.write(i)

with open('/data/model/SentencePairMatch/tmpdata/zh.train', 'w') as f:
    for i in [i[1] for i in data]:
        f.write(i)

with open('/data/model/SentencePairMatch/tmpdata/label.eval', 'w') as f:
    for i in [i[2] for i in data]:
        f.write(i + '\n')
