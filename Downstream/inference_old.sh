#!/bin/bash

#SBATCH --job-name=eval_main_prior_large_subj01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100"
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --output=./slurm-logs/%x-%j.out
#SBATCH --error=./slurm-logs/%x-%j.err

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BATCH_SIZE=32
overlay_ext3=/scratch/yz10381/singularity/fMRI.ext3
PYTHON_CMD="python recon_inference.py"
MODEL_NAME="main_prior_large"
CMD="
source /ext3/env.sh
cd ${SCRIPT_DIR}
export TAG='iter_195000'
export SAVE_DIR='tests/main_prior_large_subj02'

export SSL_CERT_FILE=/scratch/yz10381/CODES/IVP/Brain_Decoding/tmp/cacert.pem

${PYTHON_CMD} \
    --data_path=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --cache_dir=/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2 \
    --model_name=${MODEL_NAME}\
    --multi_subject="1,2,3,4,5,6,7,8" \
    --subj=2 \
    --batch_size=${BATCH_SIZE} \
    --max_lr=5e-05 \
    --mixup_pct=0.33 \
    --num_epochs=150 \
    --use_prior \
    --prior_scale=30 \
    --clip_scale=1 \
    --no-blurry_recon \
    --blur_scale=0.5 \
    --no-use_image_aug \
    --n_blocks=4 \
    --encoder_hidden_dim=320 \
    --decoder_hidden_dim=1280 \
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
    --wandb_project=main_fmri \
    --full_attention \
    --n_blocks_decoder=6 \
    --drop=0.1 \
    --progressive_dims \
    --initial_tokens=15000 \
    --dim_scale_factor=0 \
    "

echo "===executing command: $CMD"
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "$CMD"