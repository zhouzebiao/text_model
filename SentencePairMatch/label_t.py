# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020-01-28 18:
"""

with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_en.txt', 'r') as f1:
    with open('/data/D_NMT/translate_zhen/raw_data/V2.1/raw_enzh_zh.txt', 'r') as f2:
        line1 = f1.readlines()
        line2 = f2.readlines()
        line1 = [i for i in line1][20000000:30000000]
        line2 = [j for j in line2][20000001:30000001]
        le = len(line2)
        l1 = len(line1)
        print(l1, le)
        count = 0
        with open('/data/model/tmpdir/en', 'a') as f3:
            with open('/data/model/tmpdir/zh', 'a') as f4:
                with open('/data/model/tmpdir/label', 'a') as f5:
                    for index, i in enumerate(line1):
                        f3.write(i)
                        f4.write(line2[index])
                        f5.write('0\n')
                        count += 1

