import itertools
import os
from datetime import datetime
import hashlib


def generate_sbatch_fmri(
    job_name="brainnat",
    hour=2,
    minute=00,
    constraint="a100|h100",
    overlay_ext3=None,
    output_dir_base="./inference-jobs/",
    script_name="Train.py",
    num_gpus=1,
    batch_size=4,
    singularity_path="/scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif",
    project_dir="/scratch/ky2684/brain-decoding/Brain_Decoding/Downstream",
    use_env_var=False,
    env_file_path="/scratch/ky2684/brain-decoding/Brain_Decoding/.env",
    ssl_cert_file_path="/scratch/ky2684/brain-decoding/Brain_Decoding/tmp/cacert.pem",
    params=None,
):

    if params is None or not overlay_ext3:
        raise ValueError("Params cannot be None")

    # if use_env_var and not env_file_path:
    #     raise ValueError("env file path not present")

    # Add current date to job name
    current_date = datetime.now().strftime("%Y%m%d")
    job_name = f"{job_name}_{current_date}"

    # Start constructing the sbatch script
    text = "#!/bin/bash\n\n"
    text += f"#SBATCH --job-name={job_name}\n"
    text += "#SBATCH --nodes=1\n"
    text += "#SBATCH --cpus-per-task=8\n"
    text += "#SBATCH --mem=64GB\n"
    text += f"#SBATCH --time={hour}:{minute:02d}:00\n"
    text += f"#SBATCH --gres=gpu:{num_gpus}\n"
    text += f'#SBATCH --constraint="{constraint}"\n'
    text += "#SBATCH --account=pr_60_tandon_priority\n"
    text += "#SBATCH --output=./slurm-logs/%x-%j.out\n"
    text += "#SBATCH --error=./slurm-logs/%x-%j.err\n\n"

    text += f"overlay_ext3={overlay_ext3}\n"

    text += "singularity exec --nv \\\n"
    text += "    --overlay ${overlay_ext3}:ro \\\n"
    text += f"    {singularity_path} \\\n"
    text += '    /bin/bash -c "\n'
    text += "source /ext3/env.sh\n"
    text += f"cd {project_dir}\n\n"

    text += f"export SSL_CERT_FILE={ssl_cert_file_path}\n"
    text += "python inference.py\\\n"

    for param_level in params.keys():
        for name, value in params[param_level].items():
            if param_level == "base":
                text += f"    {name}={value} \\\n"
            else:
                text += f"    {param_level}.{name}={value} \\\n"

    text += '"\n'

    # Save the sbatch script to a file
    os.makedirs(output_dir_base, exist_ok=True)
    job_file = os.path.join(output_dir_base, f"{job_name}.sbatch")
    with open(job_file, "w") as f:
        f.write(text)
    print(f"sbatch {job_file}")
    return text


def generate_inference_jobs(base_params, param_ranges, job_params):
    jobs = []

    # Generate all combinations of parameter values
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    model_name_param = base_params["base"]["model_name"]
    for values in itertools.product(*param_values):
        params = base_params.copy()

        for name, value in zip(param_names, values):
            for param_level in params.keys():
                if name in params[param_level]:
                    params[param_level][name] = value

        # Generate a unique job name
        params_key = f"{'_'.join([f'{name}_{value}' for name, value in zip(param_names, values)])}"

        # Wandb has a limit of 128 characters on the name, so hashing the name
        model_name = (
            f"{model_name_param}_{hashlib.md5(params_key.encode()).hexdigest()[:10]}"
        )
        params["base"]["model_name"] = model_name

        # Generate the job script for seen subjects
        job_name = f"{params['base']['wandb_project']}_{model_name}_seen_inference"
        params["data"]["multi_subject"] = "'[2, 3, 4, 5, 6, 7, 8]'"
        job_script = generate_sbatch_fmri(
            job_name=job_name,
            params=params,
            **job_params,
        )
        jobs.append((job_name, job_script))

        # Generate the job script for unseen subjects
        job_name = f"{params['base']['wandb_project']}_{model_name}_unseen_inference"
        params["data"]["multi_subject"] = "'[1]'"
        job_script = generate_sbatch_fmri(
            job_name=job_name,
            params=params,
            **job_params,
        )
        jobs.append((job_name, job_script))

    return jobs


default_params = {
    "base": {
        "wandb_log": True,
        "wandb_project": "fmri_new",
        "wandb_entity": "nyu_brain_decoding",
        "model_name": "tomer_qformer",
    },
    "data": {
        "data_path": "/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2",
        "cache_dir": "/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2",
        "subj": 2,
        "num_sessions": 40,
        "multi_subject": "'[2, 3, 4, 5, 6, 7, 8]'",
        "new_test": True,
    },
    "model": {
        "n_blocks": 4,
        "decoder_hidden_dim": 1280,
        "encoder_hidden_dim": 256,
        "use_mixer": False,
        "num_heads": 8,
        "tome_r": 1000,
        "last_n_features": 16,
        "nat_depth": 8,
        "nat_num_neighbors": 8,
        "full_attention": True,
        "n_blocks_decoder": 6,
        "drop": 0.1,
        "progressive_dims": True,
        "initial_tokens": 15000,
        "dim_scale_factor": 0,
        "clip_seq_dim": 256,
        "clip_emb_dim": 1664,
    },
    "train": {
        "use_prior": True,
        "blurry_recon": False,
        "batch_size": 8,
        "global_batch_size": None,
        "num_epochs": 150,
        "seed": 42,
        "max_lr": 5.0e-05,
        "lr_scheduler_type": "cycle",
        "ckpt_saving": True,
        "ckpt_interval": 1,
        "ckpt_iter": 15000,
        "mixup_pct": 0.33,
        "use_image_aug": False,
        "clip_scale": 1.0,  # base task
        "blur_scale": 0.5,  # only applies when model.blurry_recon is True
        "prior_scale": 30,  # only applies when model.use_prior is True
        "multisubject_ckpt": None,
        # "use_grad_clip": True
    },
}

param_ranges = {
    "batch_size": [16],
    # "nat_depth": [8],
    # "n_blocks_decoder": [6],
    # "use_grad_clip": [True]
    # "use_siren_emb": [False],
    # "use_avg_pool": [True],
    # "mlp_clip_head": [True],
    # "use_prior": [False],
    # "clip_seq_dim": [256],
    # "clip_emb_dim": [1280],
    
    # Variable Perceiver ablations
    # "n_blocks_decoder": [8],
    # "head_dim": [128],
    # "num_heads": [8],
    # "self_per_cross_attn": [2],
    # "variable_hidden_dims": ["'[128, 185, 266, 384, 554, 800, 1154, 1664]'"],
    
    # Hierarchical Perceiver ablation parameters
    # "downsample_factors": ["'[2, 2, 2, 2]'"],
    # "downsample_method": ["grid"],  # "knn" is too slow for most purposes
    # "use_residual": [False],
    # "head_dim": [64],
    # "num_heads": [8],
    # "self_per_cross_attn": [2],
    # "n_blocks": [4],  # Number of hierarchical levels
}

job_params = {
    "hour": 1,
    "minute": 00,
    "constraint": "h100|a100",
    "num_gpus": 1,
    "batch_size": default_params["train"]["batch_size"],
    "overlay_ext3": "/scratch/ky2684/brain-decoding/fmri-img-reconstruct.ext3",
    "singularity_path": "/scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif",
    "project_dir": "/scratch/ky2684/brain-decoding/Brain_Decoding/Downstream",
    "ssl_cert_file_path": "/scratch/ky2684/brain-decoding/Brain_Decoding/tmp/cacert.pem",
}

if __name__ == "__main__":
    jobs = generate_inference_jobs(default_params, param_ranges, job_params)
