#!/bin/bash

#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100"
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --output=./slurm-logs/%x-%j.out
#SBATCH --error=./slurm-logs/%x-%j.err


overlay_ext3=/scratch/ky2684/brain-decoding/fmri-img-reconstruct.ext3
export NUM_GPUS=2  # Set to equal gres=gpu:#!
export BATCH_SIZE=16 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
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
export $(grep -v '^#' .env | xargs) && wandb login
cd /scratch/ky2684/brain-decoding/Brain_Decoding/Downstream

export SSL_CERT_FILE=/scratch/ky2684/brain-decoding/Brain_Decoding/tmp/cacert.pem
accelerate launch --num_processes=${NUM_GPUS} --main_process_port=${MASTER_PORT} --mixed_precision=fp16 Train.py\
    --data_path=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --cache_dir=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --model_name=main_no_prior_perciever \
    --multi_subject="2,3,4,5,6,7,8" \
    --subj=2 \
    --batch_size=${BATCH_SIZE} \
    --max_lr=5e-05 \
    --mixup_pct=0.33 \
    --num_epochs=100 \
    --no-use_prior \
    --prior_scale=30 \
    --clip_scale=1 \
    --no-blurry_recon \
    --blur_scale=0.5 \
    --no-use_image_aug \
    --n_blocks=4 \
    --encoder_hidden_dim=256 \
    --decoder_hidden_dim=512 \
    --num_sessions=40 \
    --ckpt_interval=3 \
    --ckpt_saving \
    --wandb_log \
    --num_heads=8 \
    --tome_r=1000 \
    --last_n_features=16 \
    --nat_depth=8 \
    --nat_num_neighbors=8 \
    --lr_scheduler_type=cycle \
    --seed=42 \
    --no-use_mixer \
    --new_test \
    --wandb_project=fmri_new \
    --full_attention \
    --n_blocks_decoder=6 \
    --drop=0.1 \
    --progressive_dims \
    --initial_tokens=15000 \
    --dim_scale_factor=0 \
"
