import os
NONE        = 0
BASE_THEORY = 1
MSE_THEORY  = 2

BASE_MODEL      = 0
LARGE_MODEL     = 1
LARGE_BI_MODEL  = 2

seed = 673511663608267506419597994679

duration_loss_scaler    = 2
step_loss_scaler        = 5

key_weight = 1
octave_weight = 1


base_command        = f"py src/main.py --mode train --model_dir ./models_final/ --mode.data ./datasets/maestro_50"
scalers_str         = f"--duration_loss_scaler {duration_loss_scaler} --step_loss_scaler {step_loss_scaler}"
seed_str            = f"--mode.fixed_seed {seed}"
loss_weights_str    = f"--mode.key_weight {key_weight} --mode.octave_weight {octave_weight}"

os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {BASE_MODEL} --mode.music_theory {NONE} --mode.name base-model")
os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {BASE_MODEL} --mode.music_theory {BASE_THEORY} --mode.name base-model-music-theory")
os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {BASE_MODEL} --mode.music_theory {MSE_THEORY} --mode.name base-model-music-theory-mse")

os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {LARGE_MODEL} --mode.music_theory {NONE} --mode.name large-model")
os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {LARGE_MODEL} --mode.music_theory {BASE_THEORY} --mode.name large-model-music-theory")
os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {LARGE_MODEL} --mode.music_theory {MSE_THEORY} --mode.name large-model-music-theory-mse")

os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {LARGE_BI_MODEL} --mode.music_theory {NONE} --mode.name large-bi-model")
os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {LARGE_BI_MODEL} --mode.music_theory {BASE_THEORY} --mode.name large-bi-model-music-theory")
os.system(f"{base_command} {scalers_str} {seed_str} {loss_weights_str} --mode.model {LARGE_BI_MODEL} --mode.music_theory {MSE_THEORY} --mode.name large-bi-model-music-theory-mse")