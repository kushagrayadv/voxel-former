# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export NUM_LOCAL_GPUS=$(nvidia-smi -L | wc -l)
export NUM_GPUS=$(($NUM_LOCAL_GPUS * $COUNT_NODE))
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}
export SSL_CERT_FILE=/scratch/ky2684/brain-decoding/Brain_Decoding/tmp/cacert.pem
accelerate launch --num_processes=${NUM_GPUS} --main_process_port=${MASTER_PORT} --mixed_precision=bf16 Train.py
# accelerate launch --num_processes=2 --main_process_port=29500 --mixed_precision=bf16 Train.py $@ # first run

# Resume running
# accelerate launch --num_processes=2 --main_process_port=29501 --mixed_precision=bf16 Train.py --config-dir ${instance_dir}/.hydra \
#     'hydra.job_logging.handlers.file.filename=train.log' \
#     "instance_dir=${instance_dir}" \
#     'hydra.run.dir=${instance_dir}' \
#     $@ # resume running