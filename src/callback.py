import tensorflow as tf

class Callback():
    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)

class Cb1(Callback):
    all_pitch_loss:     tf.float32
    all_step_loss:      tf.float32
    all_duration_loss:  tf.float32
    test_pitch_loss:    tf.float32
    test_step_loss:     tf.float32
    test_duration_loss: tf.float32
    test_list:          list
    train_list:         list
    start:              tf.float64
    def __init__(self):
        self.all_pitch_loss = 0.0
        self.all_step_loss = 0.0
        self.all_duration_loss = 0.0
        self.test_pitch_loss = 0.0
        self.test_step_loss = 0.0
        self.test_duration_loss = 0.0
        self.start = tf.cast(0.0, dtype=tf.float64)
        self.test_list = [[],[],[],[]]
        self.train_list = [[],[],[],[]]

    def call(self, i, pitch, step, duration, mode, max_step, **kwargs):
        if (mode == 'test'):
            if i == max_step:
                self.test_list[0].append((self.test_pitch_loss.numpy() + self.test_step_loss.numpy() + self.test_duration_loss.numpy()) / max_step)
                self.test_list[1].append(self.test_pitch_loss.numpy() / max_step)
                self.test_list[2].append(self.test_step_loss.numpy() / max_step)
                self.test_list[3].append(self.test_duration_loss.numpy() / max_step)
            elif i == 0:
                self.test_pitch_loss = 0.0
                self.test_step_loss = 0.0
                self.test_duration_loss = 0.0

            self.test_pitch_loss      += pitch 
            self.test_step_loss       += step 
            self.test_duration_loss   += duration 
            # Log every 100 batches.
            if i % 100 == 0:
                i = tf.cast(i+1, dtype=tf.float32)
                avg_pitch_loss = self.test_pitch_loss / i
                avg_step_loss = self.test_step_loss / i
                avg_duration_loss = self.test_duration_loss / i
                tf.print(
                    "Test loss (avg) at step ", i-1," - Loss: ", avg_pitch_loss + avg_step_loss + avg_duration_loss, " - pitch: ", avg_pitch_loss, ", step: ", avg_step_loss, ", duration: ", avg_duration_loss)
        elif (mode == 'train'):
            if i == max_step:
                self.train_list[0].append((self.all_pitch_loss.numpy() + self.all_step_loss.numpy() + self.all_duration_loss.numpy()) / max_step)
                self.train_list[1].append(self.all_pitch_loss.numpy() / max_step)
                self.train_list[2].append(self.all_step_loss.numpy() / max_step)
                self.train_list[3].append(self.all_duration_loss.numpy() / max_step)
            elif i == 0:
                self.all_pitch_loss = 0.0
                self.all_step_loss = 0.0
                self.all_duration_loss = 0.0

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
                    "Training loss (avg) at step ", i-1, " - Loss: ", avg_pitch_loss + avg_step_loss + avg_duration_loss, " - pitch: ", avg_pitch_loss, ", step: ", avg_step_loss, ", duration: ", avg_duration_loss)
