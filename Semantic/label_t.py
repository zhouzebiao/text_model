# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020-01-28 18:
"""

with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_en.txt', 'r') as f1:
    with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_zh.txt', 'r') as f2:
        line1 = f1.readlines()
        line2 = f2.readlines()
        line1 = [i.replace('\n', '').strip() for i in line1][3000000:5000000]
        line2 = [j.replace('\n', '').strip() for j in line2][3000001:5000001]
        le = len(line2)
        l1 = len(line1)
        print(l1, le)
        count = 0
        with open('/data/model/Semantic/raw_data/train', 'w') as f3:
            for i, j in zip(line1, line2):
                f3.write('%s\t%s\t%s\n' % (i, j, '0'))

with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_en.txt', 'r') as f1:
    with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_zh.txt', 'r') as f2:
        line1 = f1.readlines()
        line2 = f2.readlines()
        line1 = [i.replace('\n', '').strip() for i in line1][1000000:3000000]
        line2 = [j.replace('\n', '').strip() for j in line2][1000000:3000000]
        le = len(line2)
        l1 = len(line1)
        print(l1, le)
        count = 0
        with open('/data/model/Semantic/raw_data/train', 'a') as f3:
            for i, j in zip(line1, line2):
                f3.write('%s\t%s\t%s\n' % (i, j, '1'))
