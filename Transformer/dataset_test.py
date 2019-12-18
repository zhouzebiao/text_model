# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-11-26 17:
"""

import dataset
import tensorflow as tf


class DatasetTest(tf.test.TestCase):

    def test_input_pipeline(self):
        max_length = 24
        buckets_min, buckets_max = dataset.create_min_max_boundaries(max_length, min_boundary=4, boundary_scale=2)
        print(buckets_min, buckets_max)
        self.assertEqual(buckets_min, [0, 4, 8, 16, 24])
        self.assertEqual(buckets_max, [4, 8, 16, 24, 25])


if __name__ == "__main__":
    tf.test.main()
"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='-1' python dataset_test.py

"""
