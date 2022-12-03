from consts import *

class stepLossNoL():
    def __init__(self,  batch_size: int):
        self.batch_size = batch_size
        self.step_scale = tf.constant([0.1], dtype=tf.float32)
        self.zero = tf.fill((self.batch_size, 1), 0.0)
        self.ones = tf.fill((self.batch_size, 1), 1.0)

    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, max: tf.Tensor, min: tf.Tensor):
        #min = tf.cast(min, tf.float32)
        #max = tf.cast(min, tf.float32)
        #min = tf.reshape(min, y_pred.shape)
        #max = tf.reshape(max, y_pred.shape)
#
        #r1 = tf.math.subtract(y_pred, 0) + 1
        #r2 = tf.math.subtract(max, y_pred)   + 1   

        return mean_squared_error(y_true, y_pred) #cross_entropy_no_log(self.ones, r1)  + cross_entropy_no_log(self.ones, r2) + 1.5 * mean_squared_error(y_true, y_pred) 

class durationLossNoL():
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.ones = tf.fill((self.batch_size, 1), 1.0)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, max: tf.Tensor, min: tf.Tensor):
        #min = tf.cast(min, tf.float32)
        #max = tf.cast(min, tf.float32)
        #min = tf.reshape(min, y_pred.shape)
        #max = tf.reshape(max, y_pred.shape)
#
        #r1 = tf.math.subtract(y_pred, min) + 1
        #r2 = tf.math.subtract(max, y_pred) + 1

        return  mean_squared_error(y_true, y_pred) #cross_entropy_no_log(self.ones, r1)  + cross_entropy_no_log(self.ones, r2) + 1.5 *mean_squared_error(y_true, y_pred) 
        
class MusicLossNoL():
    key_weight:     float
    octave_weight:  float
    batch_size:     int
    def __init__(self, batch_size: int, key_weight: float, octave_weight: float):
        self.key_weight = key_weight
        self.octave_weight = octave_weight
        self.batch_size = batch_size
        self.one = tf.fill((self.batch_size, 1), 1.0)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        key = tf.cast(keys,dtype=tf.int64)
        octaves = tf.cast(tf.math.floordiv(y_true, 12), dtype=tf.int64)
        #step_1 = tf.gather(CLAMPED_IN_KEY_WEIGHTED, octave, axis=0)
        #step_2 = tf.gather(CLAMPED_IN_KEY_WEIGHTED, key, axis=1)
        step_1 = tf.gather(CLAMPED_IN_KEY, key, axis=0)
        step_2 = tf.reshape(tf.tile(step_1, [self.batch_size]), [ self.batch_size, tf.shape(step_1)[0]])
        return cross_entropy(step_2, y_pred) #+ sparse_entropy(y_true, y_pred)

class stepLoss():
    def __init__(self):
        self.step_scale = tf.constant([0.1], dtype=tf.float32)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sum: tf.Tensor):
        return mean_squared_error(y_true=y_true, y_pred=y_pred) + tf.reduce_mean(tf.multiply(sum, self.step_scale))
        
class durationLoss():
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.zero = tf.fill((self.batch_size, 1), 0.0)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, max: tf.Tensor, min: tf.Tensor):
        less = tf.cast(tf.less(y_true, min), dtype=tf.float32)
        great = tf.cast(tf.greater(y_true, max), dtype=tf.float32)
        add = tf.add(less, great)
        return mean_squared_error(y_true=y_true, y_pred=y_pred) + mean_squared_error(add, self.zero)
        
class MusicLoss():
    key_weight:     float
    octave_weight:  float
    batch_size:     int
    def __init__(self, batch_size: int, key_weight: float, octave_weight: float):
        self.key_weight = key_weight
        self.octave_weight = octave_weight
        self.batch_size = batch_size
        self.one = tf.fill((self.batch_size, 1), 1.0)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        sample_pred = tf.random.categorical(y_pred, num_samples=1)
        y = tf.math.floormod(sample_pred, 12)
        equal = tf.math.equal(y, keys)
        f = tf.reduce_any(equal, 1)
        leap_pred = tf.abs(sample_pred - tf.cast(y_true, dtype=tf.int64))
        return sparse_entropy(y_true, y_pred) + self.key_weight * cross_entropy(f, self.one) + self.octave_weight * mean_squared_error(leap_pred, self.one)

class MusicLoss2():
    key_weight:     float
    octave_weight:  float
    batch_size:     int
    def __init__(self, batch_size: int, key_weight: float, octave_weight: float):
        self.key_weight = key_weight
        self.octave_weight = octave_weight
        self.batch_size = batch_size
        self.one = tf.fill((self.batch_size, 1), 1.0)
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        #sample_pred = tf.random.categorical(y_pred, num_samples=1)
        #y = tf.math.floormod(sample_pred, 12)
        #equal = tf.math.equal(y, keys)
        #f = tf.reduce_any(equal, 1)
        #tr = tf.fill((self.batch_size, 1), 1.0)

        #true_octave = tf.math.floordiv(y_true, 12)
        #pred_octave = tf.cast(tf.math.floordiv(sample_pred, 12), dtype=tf.float32)

        #return sparse_entrophy(y_true, y_pred) + self.key_weight * mean_squared_error(f, tr) + mean_squared_error(true_octave, pred_octave)
        sample_pred = tf.random.categorical(y_pred, num_samples=1)
        y = tf.math.floormod(sample_pred, 12)
        equal = tf.math.equal(y, keys)
        f = tf.reduce_any(equal, 1)
        leap_pred = tf.abs(sample_pred - tf.cast(y_true, dtype=tf.int64))
        return sparse_entropy(y_true, y_pred) + self.key_weight * mean_squared_error(f, self.one) + self.octave_weight * mean_squared_error(leap_pred, self.one)
        
class MusicLossBasic():
    key_weight:     float
    octave_weight:  float
    batch_size: int
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.key_weight = 0
        self.octave_weight = 0
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        return sparse_entropy(y_true, y_pred) 