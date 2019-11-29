set -e
set -x
export PYTHONPATH='src'
log_dir="../saved/logs/"
cd src

experiment_index=100
log_file=$log_dir$experiment_index

python main.py \
    --experiment_index=$experiment_index \
    --cuda=0 \
    --dataset=mnist \
    --n_epochs=20 \
    --num_workers=0 \
    --eval_frequency=1 \
    --upper_threshold=0.9 \
    2>&1 | tee $log_file
