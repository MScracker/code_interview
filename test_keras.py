#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(7, 7, 1), filters=20, kernel_size=3, padding="same",
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, name='Dense')
])
model.summary()