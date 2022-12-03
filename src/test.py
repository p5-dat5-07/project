import tensorflow as tf
#@tf.function
#def shift(inputs, shift, axis, pad_val = 0.0):
#    old_shape = inputs.shape
#
#    pad_shape = list(inputs.shape)
#    input_pad = tf.fill(pad_shape, pad_val)
#    inputs = tf.concat((inputs, input_pad), axis) 
#    
#    
#    input_roll = tf.roll(inputs, shift, axis)
#    ret = tf.slice(input_roll, [0 for _ in range(len(old_shape))], old_shape)
#
#    return ret
#
#
##key = tf.constant([1,2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#a = tf.constant([0.0 ])
#b = tf.constant([0.0, 1.1, 1.1, 1.1, 0.0, 0.0])
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.4, 0.0, 1.4, 0.0, 1.4, 1.4, 0.0, 1.4, 0.0, 1.4, 0.0, 1.4, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.6, 0.0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#print(a*b)
#print(shift(key), k-4, 0)

test = tf.constant([
    [
        [1,1],
        [2,1],
        [3,1],
    ],
    [
        [4,2],
        [5,2],
        [6,2],
    ]
    ,
    [
        [7,3],
        [8,3],
        [9,3],
    ]
], dtype=tf.float32)
note_1 = tf.math.floordiv(1, 12)
note_2 = tf.math.floordiv(34, 12)
key = 2
step_1 = tf.gather(test, [note_1, note_2], axis=0)
step_2 = tf.gather(step_1, 1, axis=1)
step_3 = tf.repeat(step_2, self.batch_size, axis=2)
print(step_1)
print(step_2)
print(step_3)