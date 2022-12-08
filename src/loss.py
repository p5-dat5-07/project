from consts import *

class StepLossBasic():
    def __init__(self):
        self.step_scale = tf.constant([0.1], dtype=tf.float32)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, max: tf.Tensor, num_zero: tf.Tensor):
        return mean_squared_error(y_true,y_pred)

class StepLossSimple():

    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, max: tf.Tensor, num_zero: tf.Tensor):  
        return  mean_squared_error(y_true, tf.maximum(y_pred, 0))

class DurationLossBasic():
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, max: tf.Tensor, min: tf.Tensor):
        return mean_squared_error(y_true, y_pred)

class DurationLossSimple():
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.ones = tf.fill((self.batch_size, 1), 1.0)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, max: tf.Tensor, min: tf.Tensor):
        min = tf.cast(min, tf.float32)
        max = tf.cast(max, tf.float32)
        min = tf.reshape(min, y_pred.shape)
        max = tf.reshape(max, y_pred.shape) * 2

        r1 = tf.math.subtract(y_pred, min) + 1
        r2 = tf.math.subtract(max, y_pred) + 1

        return  (cross_entropy_no_log(self.ones, r1)  + cross_entropy_no_log(self.ones, r2)) + mean_squared_error(y_true, y_pred)
        

class MusicLossBasic():
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        return sparse_entropy(y_true, y_pred) 

class MusicLossSimple():
    key_weight:     float
    octave_weight:  float
    batch_size:     int
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        key = tf.cast(keys,dtype=tf.int64)
        predictions = tf.cast(y_true, dtype=tf.int64)
        step_1 = tf.gather(CLAMPED_IN_KEY, key, axis=0)
        step_2 = tf.gather(WEIGHT_NEXT_NOTE, predictions, axis=0)
        step_3 = tf.nn.softmax(tf.nn.relu((step_1 + step_2)))
        return categorical_cross_entropy(step_3, y_pred)

class MusicLossAdvanced():
    key_weight:     float
    octave_weight:  float
    batch_size:     int
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        key = tf.cast(keys,dtype=tf.int64)
        octaves = tf.cast(tf.math.floordiv(y_true, 12), dtype=tf.int64)
        predictions = tf.cast(y_true, dtype=tf.int64)
        step_1 = tf.gather(CLAMPED_IN_KEY_WEIGHTED, octaves, axis=0)
        step_2 = tf.gather(step_1, key, axis=1, batch_dims=1)
        step_3 = tf.gather(WEIGHT_NEXT_NOTE, predictions, axis=0)
        step_4 = tf.nn.softmax(tf.nn.relu((step_2 + step_3)))
        return categorical_cross_entropy(step_4, y_pred)