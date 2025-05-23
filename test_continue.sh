#!/bin/bash

# Load pyenv into the script (needed if running in a non-interactive shell)
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Activate the pyenv virtual environment
pyenv shell marl

run_experiment() {
    echo "Running your experiment..."
    cd /mnt/data/arl_RODE
    pwd
    output_fname="../output_logs/$(date +%Y%m%d%H%M)_output.txt"
    nohup python ./src/main.py --config=rode --env-config=sc2 with env_args.map_name=corridor n_role_clusters=3 role_interval=5 t_max=7050000 checkpoint_path="results/models/rode__2025-02-08_22-13-44" > "$output_fname" 2>&1 &
    echo "Saving to $output_fname"
    echo "Experiment completed."
}

# Call the function and wait for it to finish successfully before shutting down
# run_experiment && echo "Calling shutdown.sh..." && ./shutdown.sh
run_experiment && echo "Pretending to shutdown"