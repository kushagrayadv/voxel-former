#!/bin/bash

#SBATCH --job-name=fmri_new_single_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|a100"
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --output=./slurm-logs/%x-%j.out
#SBATCH --error=./slurm-logs/%x-%j.err

overlay_ext3=/scratch/ky2684/brain-decoding/fmri-img-reconstruct.ext3
export NUM_GPUS=1  # Set to equal gres=gpu:#!

singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    /bin/bash -c "
source /ext3/env.sh
cd /scratch/ky2684/brain-decoding/Brain_Decoding/Downstream

export SSL_CERT_FILE=/scratch/ky2684/brain-decoding/Brain_Decoding/tmp/cacert.pem
python inference.py \
    wandb_log=True \
    wandb_project=fmri_new \
    wandb_entity=nyu_brain_decoding \
    model_name=variable_perceiver_cdc568abb7 \
    data.data_path=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    data.cache_dir=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    data.subj=2 \
    data.num_sessions=40 \
    data.multi_subject='[2, 3, 4, 5, 6, 7, 8]' \
    data.new_test=True \
    model.encoder_type=linformer \
    model.decoder_type=perceiver \
    model.perceiver_type=variable \
    model.n_blocks=4 \
    model.decoder_hidden_dim=1280 \
    model.encoder_hidden_dim=256 \
    model.encoder_seq_len=2048 \
    model.share_kv=False \
    model.use_mixer=False \
    model.num_heads=8 \
    model.head_dim=64 \
    model.self_per_cross_attn=2 \
    model.tome_r=1000 \
    model.last_n_features=16 \
    model.nat_depth=8 \
    model.nat_num_neighbors=8 \
    model.full_attention=True \
    model.n_blocks_decoder=6 \
    model.drop=0.1 \
    model.progressive_dims=True \
    model.initial_tokens=15000 \
    model.dim_scale_factor=0 \
    model.clip_seq_dim=256 \
    model.clip_emb_dim=1664 \
    model.use_siren_emb=False \
    model.use_avg_pool=False \
    model.downsample_factors='[2, 2, 2, 2]' \
    model.use_residual=True \
    model.downsample_method=grid \
    model.visualize_hierarchy=True \
    model.variable_hidden_dims='[128, 256, 512, 768, 1024, 1280]' \
    train.use_prior=True \
    train.blurry_recon=False \
    train.batch_size=32 \
    train.global_batch_size=None \
    train.num_epochs=150 \
    train.seed=42 \
    train.max_lr=5e-05 \
    train.lr_scheduler_type=cycle \
    train.ckpt_saving=True \
    train.ckpt_interval=3 \
    train.ckpt_iter=15000 \
    train.mixup_pct=0.33 \
    train.use_image_aug=False \
    train.clip_scale=1.0 \
    train.blur_scale=0.5 \
    train.prior_scale=30 \
    train.multisubject_ckpt=None \
"
