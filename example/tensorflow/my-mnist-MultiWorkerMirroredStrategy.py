import tensorflow as tf
import numpy as np
import json
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
# https://stackoverflow.com/a/55318851
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_visible_devices(gpus[args.rank], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[args.rank], True)
print("my rank ", args.rank)

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # We need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

per_worker_batch_size = 64
# single_worker_dataset = mnist_dataset(per_worker_batch_size)
# single_worker_model = build_and_compile_cnn_model()
# single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

# ======================
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        # 'worker': ["10.108.236.139:12345", "10.108.236.139:23456"]
        # 'worker': ["localhost:12345", "localhost:23456"]
        'worker': ["10.108.236.139:12345"]
    },
    'task': {'type': 'worker', 'index': args.rank}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(devices=["/gpu:0"])


num_workers = 1
# num_workers = 2

per_worker_batch_size = 64
# Here the batch size scales up by number of workers since 
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
# and now this becomes 128.
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
multi_worker_model.fit(multi_worker_dataset, epochs=10, steps_per_epoch=70)
print("xxxxxxxxxxxxxxxxxxxxxxxxx all done")
