import tensorflow as tf

class Callback():
    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)

class Cb1(Callback):
    all_pitch_loss:     tf.float32
    all_step_loss:      tf.float32
    all_duration_loss:  tf.float32
    start:              tf.float64
    def __init__(self):
        self.all_pitch_loss = 0.0
        self.all_step_loss = 0.0
        self.all_duration_loss = 0.0
        self.start = tf.cast(0.0, dtype=tf.float64)

    def call(self, i, pitch, step, duration, *kwargs):
        if i == 0:
            self.all_pitch_loss = 0.0
            self.all_step_loss = 0.0
            self.all_duration_loss = 0.0
            self.start = tf.timestamp()

        self.all_pitch_loss      += pitch 
        self.all_step_loss       += step 
        self.all_duration_loss   += duration 
        # Log every 100 batches.
        if i % 100 == 0:
            i = tf.cast(i+1, dtype=tf.float32)
            avg_pitch_loss = self.all_pitch_loss / i
            avg_step_loss = self.all_step_loss / i
            avg_duration_loss = self.all_duration_loss / i
            tf.print(
                "Training loss (avg) at step ", i-1, "(", tf.math.floor((tf.timestamp() - self.start)), "s) - Loss: ", avg_pitch_loss + avg_step_loss + avg_duration_loss, " - pitch: ", avg_pitch_loss, ", step: ", avg_step_loss, ", duration: ", avg_duration_loss)
            self.start = tf.timestamp()
