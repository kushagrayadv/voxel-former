# VoxelFormer: Parameter-Efficient Multi-Subject Visual Decoding from fMRI

**A lightweight transformer architecture for cross-subject brain decoding**

![](figs/voxelformer_overview.png)<br>

## Abstract

Recent advances in fMRI-based visual decoding have enabled compelling reconstructions of perceived images. However, most approaches rely on subject-specific training, limiting scalability and practical deployment. **VoxelFormer** is a lightweight transformer architecture that enables multi-subject training for visual decoding from fMRI. VoxelFormer integrates a Token Merging Transformer (ToMer) for efficient voxel compression and a query-driven Q-Former that produces fixed-size neural representations aligned with the CLIP image embedding space.

## Key Contributions

1. **Token Merging Transformer (ToMer)**: Dynamically reduces fMRI token count via learned attention, lowering memory cost while preserving critical information
2. **Query-driven Q-Former**: Produces fixed-size latent representations enabling multi-subject training without subject-specific layers
3. **Parameter Efficiency**: Achieves competitive performance with 39M parameters (12× reduction vs MindEye2, 24× reduction vs MindEye1)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/kushagrayadv/voxel-former.git
cd voxel-former
```

2. Run setup script to install the "fmri" virtual environment:

```bash
bash setup.sh
source fmri/bin/activate
```

3. Install additional dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-deps
```

## Usage

*Note* - The configuration for the dataset paths, model parameters and training parameters are under the ``src/conf`` directory with respective sub-directories. 

### Training

```bash
# Multi-subject training on NSD subjects 2-7
python src/Train.py \
    --model.encoder_hidden_dim=1024 \
    --model.decoder_hidden_dim=4096 \
    --model.nat_depth=4 \
    --model.num_heads=8 \
    --model.nat_num_neighbors=5 \
    --train.batch_size=24 \
    --train.max_lr=3e-4 \
    --train.num_epochs=150 \
    --data.multi_subject=[2,3,4,5,6,7]
```

### Inference

```bash
# Run image retrieval inference
python src/inference.py
```

### SLURM Jobs
The files `voxel-former-slurm-train.sbatch` and `voxel-former-slurm-inference.sbatch` provide example SLURM job scripts for training and running inference with the model in HPC environments.

## Dataset

VoxelFormer is evaluated on the **7T Natural Scenes Dataset (NSD)**:
- 8 subjects with high-resolution whole-brain fMRI
- 30-40 sessions per subject viewing natural scene images from MS-COCO
- Training: Subjects 2-7 (multi-subject training)
- Evaluation: Standard top-1 retrieval with 300-image candidate pool

## Data Requirements

### Input Format
- **fMRI Voxels**: 3D brain activation patterns from visual cortex (~10K-20K voxels per subject)
- **Coordinates**: 3D spatial coordinates for each voxel (x, y, z)
- **Images**: Natural scene stimuli from MS-COCO dataset
- **CLIP Features**: Precomputed CLIP embeddings (optional for faster training)

### Expected Data Structure
```
data/
├── nsd_data/
│   ├── voxels/              # Subject-specific voxel activations
│   ├── coordinates/         # 3D voxel coordinate mappings
│   ├── images/              # MS-COCO stimulus images  
│   └── clip_embeddings/     # Precomputed CLIP features
├── subjects/
│   ├── subj02/
│   ├── subj03/
│   └── ...
└── metadata/
    ├── sessions.json
    └── stimulus_mapping.json
```

## HPC Usage

### Environment Setup
```bash
# Interactive A100 session
srun -t 4:00:00 --mem=64000 --gres=gpu:a100:1 --pty /bin/bash
```

### Singularity Container
```bash
singularity exec --nv \
    --overlay /scratch/overlay.ext3:ro \
    /scratch/containers/pytorch_22.08.sif \
    /bin/bash -c "source /ext3/env.sh && python src/Train.py [args]"
```

## Acknowledgments

- Built upon the **Natural Scenes Dataset (NSD)** from Allen et al.
- Incorporates **CLIP** vision-language representations from OpenAI
- Inspired by **MindEye** and **MindEye2** project and repo for brain decoding
- Uses **Token Merging (ToMe)** techniques for efficient transformer computation
