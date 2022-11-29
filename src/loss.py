from consts import *

class MusicLoss():
    key_weight:     float
    octave_weight:  float
    batch_size:     int
    def __init__(self, batch_size: int, key_weight: float, octave_weight: float):
        self.key_weight = key_weight
        self.octave_weight = key_weight
        self.batch_size = batch_size
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        sample_pred = tf.random.categorical(y_pred, num_samples=1)
        y = tf.math.floormod(sample_pred, 12)
        equal = tf.math.equal(y, keys)
        f = tf.reduce_any(equal, 1)
        tr = tf.fill((self.batch_size, 1), 1.0)

        true_octave = tf.math.floordiv(y_true, 12)
        pred_octave = tf.cast(tf.math.floordiv(sample_pred, 12), dtype=tf.float32)

        return sparse_entrophy(y_true, y_pred) + self.key_weight * cross_entropy(f, tr) + mean_squared_error(true_octave, pred_octave)

class MusicLoss2():
    key_weight:     float
    octave_weight:  float
    batch_size:     int
    def __init__(self, batch_size: int, key_weight: float, octave_weight: float):
        self.key_weight = key_weight
        self.octave_weight = key_weight
        self.batch_size = batch_size
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        sample_pred = tf.random.categorical(y_pred, num_samples=1)
        y = tf.math.floormod(sample_pred, 12)
        equal = tf.math.equal(y, keys)
        f = tf.reduce_any(equal, 1)
        tr = tf.fill((self.batch_size, 1), 1.0)

        true_octave = tf.math.floordiv(y_true, 12)
        pred_octave = tf.cast(tf.math.floordiv(sample_pred, 12), dtype=tf.float32)

        return sparse_entrophy(y_true, y_pred) + self.key_weight * mean_squared_error(f, tr) + mean_squared_error(true_octave, pred_octave)
        
class MusicLossBasic():
    batch_size: int
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, keys: tf.Tensor):
        return sparse_entrophy(y_true, y_pred) 