#!/bin/bash

#SBATCH --job-name=voxel_former_slurm_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|a100"
#SBATCH --output=./slurm-logs/%x-%j.out
#SBATCH --error=./slurm-logs/%x-%j.err

overlay_ext3=/scratch/overlay.ext3
export NUM_GPUS=2  # Set to equal gres=gpu:#!
export BATCH_SIZE=32 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
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
    /scratch/containers/pytorch_22.08.sif \
    /bin/bash -c "
source /ext3/env.sh
cd /path/to/project/dir/src

export SSL_CERT_FILE=/path/to/cacert.pem
accelerate launch --num_processes=${NUM_GPUS} --main_process_port=${MASTER_PORT} --mixed_precision=fp16 Train.py\
    wandb_log=True \
    wandb_project=your_project_name \
    wandb_entity=your_entity_name \
    model_name=your_model_name \
    data.data_path=/path/to/data \
    data.cache_dir=/path/to/cache \
    data.subj=2 \
    data.num_sessions=40 \
    data.multi_subject='[2, 3, 4, 5, 6, 7, 8]' \
    data.new_test=True \
    model.n_blocks=4 \
    model.decoder_hidden_dim=1280 \
    model.encoder_hidden_dim=256 \
    model.num_heads=8 \
    model.tome_r=1000 \
    model.last_n_features=16 \
    model.nat_depth=2 \
    model.nat_num_neighbors=8 \
    model.full_attention=True \
    model.n_blocks_decoder=2 \
    model.drop=0.1 \
    model.progressive_dims=True \
    model.initial_tokens=15000 \
    model.dim_scale_factor=0 \
    model.clip_seq_dim=256 \
    model.clip_emb_dim=1664 \
    train.use_prior=True \
    train.blurry_recon=False \
    train.batch_size=32 \
    train.global_batch_size=None \
    train.num_epochs=150 \
    train.seed=42 \
    train.max_lr=5e-05 \
    train.lr_scheduler_type=cycle \
    train.use_grad_clip=True \
    train.ckpt_saving=True \
    train.ckpt_interval=1 \
    train.ckpt_iter=15000 \
    train.mixup_pct=0.33 \
    train.use_image_aug=False \
    train.clip_scale=1.0 \
    train.blur_scale=0.5 \
    train.prior_scale=30 \
    train.multisubject_ckpt=None \
"
