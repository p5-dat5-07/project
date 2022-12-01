import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import pathlib
from consts import *
from model import Model
from callback import Cb1
from loss import MusicLoss, MusicLoss2, MusicLossBasic, stepLoss, durationLoss, stepLossNoL, durationLossNoL, MusicLossNoL
from args import parser, Train, Generate, Data, Base
from data_manager import DataManager


def main():
    args = parser.parse_args()
    params = args.params
    mode = args.mode.mode

    if type(mode) is Train:
        mode: Train = mode
        music_loss: Loss
        step_loss: Loss
        duration_loss: Loss
        if mode.music_theory == 0:
            music_loss = MusicLossBasic(params.batch_size)
            step_loss = stepLoss()
            duration_loss = durationLoss(params.batch_size)
        elif mode.music_theory == 1:
            step_loss = stepLoss()
            duration_loss = durationLoss(params.batch_size)
            music_loss = MusicLoss(params.batch_size, mode.key_weight, mode.octave_weight)
        elif mode.music_theory == 2:
            step_loss = stepLoss()
            duration_loss = durationLoss(params.batch_size)
            music_loss = MusicLoss2(params.batch_size, mode.key_weight, mode.octave_weight)
        elif mode.music_theory == 3:
            step_loss = stepLossNoL()
            duration_loss = durationLossNoL(params.batch_size)
            music_loss = MusicLossNoL(params.batch_size, mode.key_weight, mode.octave_weight)
        else:
            raise Exception("Invalid music theory number")

        model = Model(
            params=params, pitch_loss=music_loss,
            step_loss=step_loss, duration_loss=duration_loss,
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
            fixed_seed=mode.fixed_seed
        )
        model.create_model(mode.model)
        model.summary()
        model.load_dataset(mode.data, mode.sample_dir)
        model.train_model(mode.name, mode.model_dir, mode.sample_dir, mode.save, callback = Cb1())
    elif type(mode) is Data: 
        mode: Data = mode
        dm = DataManager(params, mode)
        dm.generate_dataset()
    elif type(mode) is Generate:
        mode: Generate = mode
        model = Model(
            params=params, pitch_loss=MusicLoss(params.batch_size),
            step_loss=mean_squared_error, duration_loss=mean_squared_error,
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
            fixed_seed=mode.fixed_seed
        )
        model.load(mode.name)
        if os.path.isdir(mode.input):
            files = glob.glob(str(pathlib.Path(mode.input)/"**/*.mid*"), recursive=True)
            if not os.path.exists(mode.output):
                os.mkdir(mode.output)
            for file in files:
                for i in range(0, mode.amount):
                    model.generate_notes(file, f"{mode.output}/{i}-{os.path.basename(file)}")
        elif mode.amount > 1:
            if not os.path.exists(mode.output):
                os.mkdir(mode.output)
            for i in range(0, mode.amount):
                model.generate_notes(mode.input, f"{mode.output}/{i}-{os.path.basename(mode.input)}")
        else:
            model.generate_notes(mode.input, mode.output)

main()