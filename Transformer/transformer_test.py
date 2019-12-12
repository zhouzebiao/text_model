import tensorflow as tf
import transformer
from official.transformer.model import model_params


class TransformerV2Test(tf.test.TestCase):

    def setUp(self):
        self.params = params = model_params.TINY_PARAMS
        params["batch_size"] = params["default_batch_size"] = 16
        params["use_synthetic_data"] = True
        params["hidden_size"] = 128
        params["embedding_size"] = 128
        params["num_hidden_layers"] = 2
        params["filter_size"] = 14
        params["num_heads"] = 8
        params["vocab_size"] = 41
        params["extra_decode_length"] = 2
        params["beam_size"] = 3
        params["data_type"] = tf.float32
        params["train"] = True
        params["layer_postprocess_dropout"] = 0.1

    def test1_create_model_train(self):
        model = transformer.create_model(self.params)
        inputs, outputs = model.inputs, model.outputs
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(inputs[1].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None, 41])
        self.assertEqual(outputs[0].dtype, tf.float32)

    def test1_create_model_not_train(self):
        self.params["train"] = False
        model = transformer.create_model(self.params)
        inputs, outputs = model.inputs, model.outputs
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None])
        self.assertEqual(outputs[0].dtype, tf.int32)
        self.assertEqual(outputs[1].shape.as_list(), [None])
        self.assertEqual(outputs[1].dtype, tf.float32)


if __name__ == "__main__":
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='-1' python transformer_test.py
"""
