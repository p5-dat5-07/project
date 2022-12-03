import tensorflow as tf

KEYS = tf.constant([
    # Major Keys
    [0, 2, 4, 5, 7, 9, 11],
    [1, 3, 5, 6, 8, 10, 0],
    [2, 4, 6, 7, 9, 11, 1],
    [3, 5, 7, 8, 10, 0, 2],
    [4, 6, 8, 9, 11, 1, 3],
    [5, 7, 9, 10, 0, 2, 4],
    [6, 8, 10, 11, 1, 3, 5],
    [7, 9, 11, 0, 2, 4, 6],
    [8, 10, 0, 1, 3, 5, 7],
    [9, 11, 1, 2, 4, 6, 8],
    [10, 0, 2, 3, 5, 7, 9],
    [11, 1, 3, 4, 6, 8, 10],
    # Minor Keys
    [0, 2, 3, 5, 7, 8, 10],
    [1, 3, 4, 6, 8, 9, 11],
    [2, 4, 5, 7, 9, 10, 0],
    [3, 5, 6, 8, 10, 11, 1],
    [4, 6, 7, 9, 11, 0, 2],
    [5, 7, 8, 10, 0, 1, 3],
    [6, 8, 9, 11, 1, 2, 4],
    [7, 9, 10, 0, 2, 3, 5],
    [8, 10, 11, 1, 3, 4, 6],
    [9, 11, 0, 2, 4, 5, 7],
    [10, 0, 1, 3, 5, 6, 8],
    [11, 1, 2, 4, 6, 7, 9],
], dtype=tf.int64)

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_clamped_keys():
    res = []
    for i in range(0, 24):
        res.append([])
        for note in range(0, 128):
            # Subtract 1 since it is zero based indexing
            if note >= 20 and note <= 107:
                res[i].append(1)
            else:
                res[i].append(0)
    return res

def ensure_in_key(keys):
    res = get_clamped_keys()
    for key in range(len(keys)):
        for note in range(0,128):
            if note % 12 not in keys[key]:
                res[key][note] = 0
    return res

def calculate_weighted_keys(keys):
    offset = 0.5
    res = []
    for octave in range(11):
        oct = []
        for key in keys:
            t_key = []
            for i in range(len(key)):
                key_octave = math.floor(i / 12)
                diff = abs(key_octave - octave)
                if diff == 0:
                    t_key.append(key[i])
                elif key[i] == 1 and diff !=0:
                    t_key.append(round(sigmoid(-diff + offset), 3))
                else:
                    t_key.append(0)
            oct.append(t_key)
        res.append(oct)
    return res


CLAMPED_KEYS = get_clamped_keys()
CLAMPED_IN_KEY = ensure_in_key(KEYS)
CLAMPED_IN_KEY_WEIGHTED = calculate_weighted_keys(CLAMPED_IN_KEY)
# Tensorflow consts
Loss = tf.keras.losses.Loss
Optimizer = tf.keras.optimizers.Optimizer
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy_no_log = tf.keras.losses.BinaryCrossentropy(from_logits=False)

sparse_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True) 
mean_squared_error = tf.keras.losses.MeanSquaredError() 
