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

major_chords = [
  [0, 4, 7, 11],
  [2, 5, 9, 12],
  [4, 7, 11, 14],
  [5, 9, 12, 16],
  [7, 11, 14, 17],
  [9, 12, 16, 19],
  [11, 14, 17, 21],
]

minor_chords = [
    [0, 3, 7, 10], 
    [2, 5, 8, 12],
    [3, 7, 10, 14],
    [5, 8, 12, 15], 
    [7, 10, 14, 17],
    [8, 12, 15, 19],
    [10, 14, 17, 20], 
]

loss_constant = 0.2

# Tensorflow consts
Loss = tf.keras.losses.Loss
Optimizer = tf.keras.optimizers.Optimizer
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
sparse_entrophy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True) 
mean_squared_error = tf.keras.losses.MeanSquaredError() 
