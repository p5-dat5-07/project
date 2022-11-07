from consts import *
from model import Params, Model
from callback import Cb1
from loss import MusicLoss

def main():
    params = Params()
    model = Model(params=params, pitch_loss=MusicLoss(params.batch_size),
            step_loss=mean_squared_error, duration_loss=mean_squared_error,
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate))
    model.create_model()
    model.summary()
    model.train_model("guud_model_yes", callback = Cb1())

main()