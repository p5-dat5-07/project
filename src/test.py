import tensorflow as tf
import json
KEYS = [
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
]
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

def weight_next_note(weight):
    res = []
    for k in range(0,128):
        res.append([0]*128)
        res[k][k] = weight
    return res
@tf.function
def rescale(X, a=0, b=1):
  repeat = X.shape[1]
  xmin = tf.repeat(tf.reshape(tf.math.reduce_min(X, axis=1), shape=[-1,1]), repeats=repeat, axis=1)
  xmax = tf.repeat(tf.reshape(tf.math.reduce_max(X, axis=1), shape=[-1,1]), repeats=repeat, axis=1)
  X = (X - xmin) / (xmax-xmin)
  return X * (b - a) + a

CLAMPED_KEYS = tf.constant(get_clamped_keys(), dtype=tf.float64)
CLAMPED_IN_KEY = tf.constant(ensure_in_key(KEYS), dtype=tf.float64)
CLAMPED_IN_KEY_WEIGHTED = tf.constant(calculate_weighted_keys(ensure_in_key(KEYS)), dtype=tf.float64)
WEIGHT_NEXT_NOTE = tf.constant(weight_next_note(1), dtype=tf.float64)

octaves = tf.cast([8]*50,dtype=tf.int64)
key = tf.cast([1]*50,dtype=tf.int64)
predictions = tf.cast([96]*50, dtype=tf.int64)

octaves_1 = tf.cast([4]*50,dtype=tf.int64)
key_2 = tf.cast([2]*50,dtype=tf.int64)
predictions_3 = tf.cast([74]*50, dtype=tf.int64)

step_1 = tf.gather(CLAMPED_IN_KEY_WEIGHTED, octaves, axis=0)
step_2 = tf.gather(step_1, key, axis=1, batch_dims=1)
step_3 = step_2 + tf.gather(WEIGHT_NEXT_NOTE, predictions, axis=0)
step_4 = tf.nn.softmax(tf.nn.relu(step_3)) # normalize vector

step_1_1 = tf.gather(CLAMPED_IN_KEY_WEIGHTED, octaves_1, axis=0)
step_2_2 = tf.gather(step_1_1, key_2, axis=1, batch_dims=1)
step_3_3 = step_2_2 + tf.gather(WEIGHT_NEXT_NOTE, predictions_3, axis=0)
step_4_4 = tf.nn.softmax(tf.nn.relu(step_3_3)) # normalize vector
x = tf.nn.softmax(tf.nn.relu(tf.constant([
    [0.25, 0.0, 0.25, 0.25]
], dtype=tf.float32)))

y = tf.constant([
    [0.25, 0.0, 0.25, 0.25]
], dtype=tf.float32)

# Zero is no good soooo add epsilojh
@tf.function
def true_cross_entropy(y_true, y_pred):
    return -tf.reduce_sum((y_true) * tf.nn.log_softmax(y_pred, axis=1), axis=1)
print(true_cross_entropy(x,y))
#print(step_1)
#print(step_2)
#print(step_3)
#print(step_4)
#
#print(rescale(x))
#print(rescale(x))
#import json
#with open('filename1.json', 'w') as f:
#    f.write(json.dumps({"a":step_4.numpy().tolist()}, indent=4))
#x = tf.random.uniform((50,), -10, 10)
#print(tf.maximum(x, 0))