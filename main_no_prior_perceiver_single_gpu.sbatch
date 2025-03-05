#!/bin/bash

#SBATCH --job-name=fmri_perceiver_single_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --output=./slurm-logs/%x-%j.out
#SBATCH --error=./slurm-logs/%x-%j.err


overlay_ext3=/scratch/ky2684/brain-decoding/fmri-img-reconstruct.ext3
export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=4 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
export $(grep -v '^#' /scratch/ky2684/brain-decoding/Brain_Decoding/.env | xargs)
cd /scratch/ky2684/brain-decoding/Brain_Decoding/Downstream
wandb login


export SSL_CERT_FILE=/scratch/ky2684/brain-decoding/Brain_Decoding/tmp/cacert.pem
accelerate launch --num_processes=${NUM_GPUS} --main_process_port=${MASTER_PORT} --mixed_precision=fp16 Train.py train.batch_size=${BATCH_SIZE}
"
