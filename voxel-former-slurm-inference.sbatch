#!/bin/bash

#SBATCH --job-name=voxel_former_slurm_inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|a100"
#SBATCH --output=./slurm-logs/%x-%j.out
#SBATCH --error=./slurm-logs/%x-%j.err

overlay_ext3=/scratch/overlay.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/containers/pytorch_22.08.sif \
    /bin/bash -c "
source /ext3/env.sh
cd /path/to/project/dir

export SSL_CERT_FILE=/path/to/cacert.pem
python src/inference.py\
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
