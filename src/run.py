import os
LOKE  = True
MÆRSK = False
SIGNE = False

BASIC       = 0 
SIMPLE      = 1
ADVANCED    = 2

BASE            = 0
LARGE           = 1
BIDIRECTIONAL   = 2

seed = 4204267269

duration_loss_scaler    = 2
step_loss_scaler        = 5

key_weight = 1
octave_weight = 1

epochs_between_samples = 5
epochs = 20

model_dir   = "./final_models2/"
data        = "./datasets/ffds400"
base_command_graph  = f"py src/graph.py --save true"
graph_format        = f"--format svg"
graph_dir           = f"--models_dir {model_dir}"
base_command        = f"py src/main.py --mode train --mode.model_dir {model_dir} --mode.data {data}"
scalers_str         = f"--duration_loss_scaler {duration_loss_scaler} --step_loss_scaler {step_loss_scaler}"
seed_str            = f"--mode.fixed_seed {seed}"
epoch_str           = f"--epochs {epochs} --epochs_between_samples {epochs_between_samples}"

if MÆRSK:
    os.system(f"{base_command} {scalers_str} {seed_str} {epoch_str} --mode.model {BASE} --mode.music_theory {SIMPLE} --mode.name base-simple")
    os.system(f"{base_command} {scalers_str} {seed_str} {epoch_str} --mode.model {BASE} --mode.music_theory {ADVANCED} --mode.name base-advanced")

    os.system(f"{base_command_graph} {graph_format} {graph_dir} --model base-simple")
    os.system(f"{base_command_graph} {graph_format} {graph_dir} --model base-advanced")

if SIGNE:
    os.system(f"{base_command} {scalers_str} {seed_str} {epoch_str} --mode.model {BASE} --mode.music_theory {BASIC} --mode.name base-basic")
    os.system(f"{base_command} {scalers_str} {seed_str} {epoch_str} --mode.model {LARGE} --mode.music_theory {ADVANCED} --mode.name large-advanced")
    

if LOKE:
    os.system(f"{base_command} {scalers_str} {seed_str} {epoch_str} --mode.model {LARGE} --mode.music_theory {BASIC} --mode.name large-basic")
    os.system(f"{base_command} {scalers_str} {seed_str} {epoch_str} --mode.model {LARGE} --mode.music_theory {SIMPLE} --mode.name large-simple")
    os.system(f"{base_command_graph} {graph_format} {graph_dir} --model large-basic")
    os.system(f"{base_command_graph} {graph_format} {graph_dir} --model large-simple")

