# -*- coding: utf-8 -*-

import os

import tensorflow as tf

output = os.popen('lspci | grep -i vga')
print(output.read())
print(tf.__version__)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import subprocess

p = subprocess.Popen('nvidia-smi', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout.readlines():
    print(line)

assert (tf.test.is_gpu_available())

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)  # Start with XLA disabled.


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 256
    x_test = x_test.astype('float32') / 256

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return ((x_train, y_train), (x_test, y_test))


(x_train, y_train), (x_test, y_test) = load_data()

"""We define the model, adapted from the Keras [CIFAR-10 example](https://keras.io/examples/cifar10_cnn/):"""


def generate_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax')
    ])


model = generate_model()

"""We train the model using the
[RMSprop](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)
optimizer:
"""


# Commented out IPython magic to ensure Python compatibility.
def compile_model(model):
    opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-7)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


model = compile_model(model)


def train_model(model, x_train, y_train, x_test, y_test, epochs=25):
    import time
    start = time.time()
    model.fit(x_train, y_train, batch_size=4096, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
    print(time.time() - start)


def warmup(model, x_train, y_train, x_test, y_test):
    # Warm up the JIT, we do not wish to measure the compilation time.
    initial_weights = model.get_weights()
    train_model(model, x_train, y_train, x_test, y_test, epochs=1)
    model.set_weights(initial_weights)


warmup(model, x_train, y_train, x_test, y_test)
train_model(model, x_train, y_train, x_test, y_test)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

"""Now let's train the model again, using the XLA compiler.
To enable the compiler in the middle of the application, we need to reset the Keras session.
"""

# Commented out IPython magic to ensure Python compatibility.
tf.keras.backend.clear_session()  # We need to clear the session to enable JIT in the middle of the program.
tf.config.optimizer.set_jit(True)  # Enable XLA.
model = compile_model(generate_model())
(x_train, y_train), (x_test, y_test) = load_data()

warmup(model, x_train, y_train, x_test, y_test)
train_model(model, x_train, y_train, x_test, y_test)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
"""On a machine with a Titan V GPU and an Intel Xeon E5-2690 CPU the speed up is ~1.17x."""

"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='1' python cift_with_xla.py

"""
