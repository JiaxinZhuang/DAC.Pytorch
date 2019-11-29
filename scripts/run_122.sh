#!/bin/bash
# Program:
#   Run main.py with specific arguments in root path

set -e # Exit immediately if a pipeline returns non-zeros signal
set -x # Print a trace of simple command

export PYTHONPATH='src'
log_dir="../saved/logs/"
cd src

# non-directory portion of the name of the shell scirpts
file_name=`basename $0`
# ##-> Deletes longest match of (*_) from front of $file_name.
experiment_index=${file_name##*_}
# %%-> Deletes longest match of $s (.*) from back of $experiment_index.
experiment_index=${experiment_index%%.*}
log_file=$log_dir$experiment_index

python main.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=mnist \
    --n_epochs=200 \
    --num_workers=0 \
    --eval_frequency=1 \
    --upper_threshold=0.9 \
    --seed=-1 \
    --track_running_stats=False \
    2>&1 | tee $log_file
