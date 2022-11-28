from consts import *

class MusicLoss():
    key_weight: float
    batch_size: int
    def __init__(self, batch_size, key_weight = 0.5):
        self.key_weight = key_weight
        self.batch_size = batch_size
    
    #@tf.function
    def __call__(self, y_true, y_pred, keys: tf.Tensor):
        return self.actual_function(y_true, y_pred, keys)
    def actual_function(self, y_true, y_pred, keys: tf.Tensor):
        index = tf.random.categorical(y_pred, num_samples=1)
        y = tf.math.floormod(index, 12)
        equal = tf.math.equal(y, keys)
        f = tf.reduce_any(equal, 1)
        tr = tf.fill((self.batch_size, 1), 1.0)
        true_octave = tf.fill((self.batch_size, 1), (tf.math.floormod(y_true[self.batch_size-1], 12)))
        pred_octave = tf.math.floormod(index, 12)

        true_octave = tf.cast(true_octave, dtype=tf.float32)
        pred_octave = tf.cast(pred_octave, dtype=tf.float32)
        test = sparse_entrophy(true_octave, pred_octave)
        breakpoint()
        return sparse_entrophy(y_true, y_pred) + self.key_weight * cross_entropy(f, tr) + sparse_entrophy(true_octave, pred_octave)
        
class MusicLossBasic():
    key_weight: float
    batch_size: int
    def __init__(self, batch_size, key_weight = 0.5):
        self.key_weight = key_weight
        self.batch_size = batch_size
    
    @tf.function
    def __call__(self, y_true, y_pred, keys: tf.Tensor):
        return sparse_entrophy(y_true, y_pred) 