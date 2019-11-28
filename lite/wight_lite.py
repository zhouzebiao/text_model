# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-11-25 10:
"""
saved_model_dir = '/data/D_NMT/translate_zhen/export/V2.1/930100_steps/1568600162/'
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.target_spec = [tf.lite.OpsSet.TFLITE_BUILTINS]
# converter.target_spec = [tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_ops=['ABS', 'ADD', 'CAST', 'CONCATENATION', 'COS', 'DIV', 'EQUAL', 'EXP', 'EXPAND_DIMS',
#                                      'FILL', 'FLOOR_DIV', 'FLOOR_MOD', 'FULLY_CONNECTED', 'GATHER', 'GATHER_ND',
#                                      'GREATER', 'LESS', 'LOG', 'LOGICAL_AND', 'LOGICAL_NOT', 'MEAN', 'MUL', 'NOT_EQUAL',
#                                      'PACK', 'POW', 'RANGE', 'REDUCE_ANY', 'REDUCE_MAX', 'RESHAPE', 'RSQRT', 'SELECT',
#                                      'SHAPE', 'SIN', 'SOFTMAX', 'SQUARED_DIFFERENCE', 'SQUEEZE', 'STRIDED_SLICE',
#                                      'SUB', 'SUM', 'TILE', 'TOPK_V2', 'TRANSPOSE', 'WHERE', 'ZEROS_LIKE']
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
# ABS, ADD, CAST, CONCATENATION, COS, DIV, EQUAL, EXP, EXPAND_DIMS, FILL, FLOOR_DIV, FLOOR_MOD, FULLY_CONNECTED, GATHER, GATHER_ND, GREATER, LESS, LOG, LOGICAL_AND, LOGICAL_NOT, MEAN, MUL, NOT_EQUAL, PACK, POW, RANGE, REDUCE_ANY, REDUCE_MAX, RESHAPE, RSQRT, SELECT, SHAPE, SIN, SOFTMAX, SQUARED_DIFFERENCE, SQUEEZE, STRIDED_SLICE, SUB, SUM, TILE, TOPK_V2, TRANSPOSE, WHERE, ZEROS_LIKE.
