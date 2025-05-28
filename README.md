# VoxelFormer: Parameter-Efficient Multi-Subject Visual Decoding from fMRI

**A lightweight transformer architecture for cross-subject brain decoding**

*Le et al., New York University Tandon School of Engineering*

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

## Architecture Overview

VoxelFormer consists of two main components working in sequence:

### 1. ToMer (Token Merging Transformer) Encoder
- **Tokenization**: Uses 1×1 convolutional layer to tokenize input voxel data
- **Positional Encoding**: SIREN-based positional embeddings from 3D voxel coordinates
- **Self-Attention**: Captures relationships among voxel tokens
- **Token Merging**: Dynamically merges highly correlated tokens during training and inference
- **Progressive Compression**: Multiple ToMer blocks progressively reduce token count

### 2. Q-Former (Query-Former) 
- **Fixed-size Output**: Produces consistent representations regardless of input voxel count
- **Cross-Attention**: Learnable query tokens attend to compressed ToMer features
- **Multi-Subject Training**: Enables training across subjects without subject-specific layers
- **CLIP Alignment**: Aligns brain representations with CLIP image embedding space

### 3. Dual-Pathway Decoding
- **Prior Transformer**: MSE loss alignment with CLIP embeddings for potential image generation
- **MLP Projector**: Contrastive learning for robust image retrieval via nearest-neighbor search

## Usage

### Training

```bash
# Multi-subject training on NSD subjects 2-7
python Downstream/Train.py \
    --model.encoder_hidden_dim=1024 \
    --model.decoder_hidden_dim=4096 \
    --model.nat_depth=4 \
    --model.num_heads=8 \
    --model.nat_num_neighbors=5 \
    --train.batch_size=24 \
    --train.max_lr=3e-4 \
    --train.num_epochs=150 \
    --train.subjects=[2,3,4,5,6,7]
```

### Inference

```bash
# Run image retrieval inference
python Downstream/recon_inference.py \
    --model_path=path/to/trained/model \
    --subject_id=2 \
    --retrieval_pool_size=300
```

### SLURM Job Generation

```bash
# Generate training jobs for HPC
python slurm_job_generator.py

# Generate inference jobs
python slurm_inference_job_generator.py
```

## Model Configuration

### ToMer Encoder Parameters
- `encoder_hidden_dim`: Base embedding dimension (default: 1024)
- `nat_depth`: Number of ToMer layers (default: 4) 
- `num_heads`: Multi-head attention heads (default: 8)
- `tome_r`: Token merging reduction rate (default: 1500-2000)
- `coord_dim`: Coordinate dimensionality for positional encoding (default: 3)

### Q-Former Parameters
- `decoder_hidden_dim`: Q-Former hidden dimension (default: 4096)
- `clip_emb_dim`: CLIP embedding dimension (default: 768)
- `clip_seq_dim`: Output sequence length (default: 256)
- `n_blocks_decoder`: Number of Q-Former blocks (default: 4)

### Training Parameters
- `lambda_MSE`: MSE loss weight (default: 30.0)
- `lambda_contrastive`: Contrastive loss weight (default: 1.0)
- `mixup_pct`: Epochs using BiMixCo loss (default: 1/3)
- `batch_size`: Training batch size (default: 24)

## Loss Functions

VoxelFormer uses a dual-pathway training strategy with phase-scheduled losses:

### Phase 1 (First 1/3 epochs): BiMixCo + MSE
- **MSE Loss**: Aligns prior embeddings with CLIP features
- **BiMixCo Loss**: InfoNCE contrastive loss with Mixup augmentation

### Phase 2 (Remaining 2/3 epochs): SoftCLIP + MSE  
- **SoftCLIP Loss**: Soft alignment with CLIP embedding distributions
- **MSE Loss**: Continued alignment for prior pathway

**Total Loss**:
```
L_total = λ_MSE × L_MSE + λ_contrastive × L_contrastive
```

## Dataset

VoxelFormer is evaluated on the **7T Natural Scenes Dataset (NSD)**:
- 8 subjects with high-resolution whole-brain fMRI
- 30-40 sessions per subject viewing natural scene images from MS-COCO
- Training: Subjects 2-7 (multi-subject training)
- Evaluation: Standard top-1 retrieval with 300-image candidate pool

## Performance Results

### Retrieval Accuracy by Subject

| Subject | Method | Forward Acc (%) | Backward Acc (%) |
|---------|--------|-----------------|------------------|
| 2 | MindEye1 | 97.1 | 93.9 |
| 2 | MindEye2 | 99.88 | 99.84 |
| 2 | **VoxelFormer** | **86.54** | **85.78** |
| 3 | MindEye1 | 90.7 | 85.7 |
| 3 | **VoxelFormer** | **74.97** | **74.17** |
| 4 | MindEye1 | 89.4 | 85.9 |
| 4 | **VoxelFormer** | **75.15** | **73.36** |
| 5 | MindEye2 | 98.39 | 96.94 |
| 5 | **VoxelFormer** | **73.03** | **71.62** |
| 6 | **VoxelFormer** | **74.93** | **74.16** |
| 7 | MindEye2 | 96.89 | 96.53 |
| 7 | **VoxelFormer** | **68.65** | **67.46** |

### Parameter Efficiency Comparison

| Method | Forward Acc (%) | Backward Acc (%) | Parameters |
|--------|-----------------|------------------|------------|
| MindEye1 | 93.6 | 90.1 | **940M** |
| MindEye2 | **98.3** | **98.3** | **469M** |
| Brain Diffuser | 21.1 | 30.3 | -- |
| **VoxelFormer** | **74.3** | **73.1** | **39M** |

**Key Insights**:
- **12× fewer parameters** than MindEye2 (39M vs 469M)
- **24× fewer parameters** than MindEye1 (39M vs 940M)  
- Consistent **>66% accuracy** across all training subjects
- **No subject-specific layers** required

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

## Performance Optimization

### Memory Efficiency
- **Token Merging**: Reduces sequence length from ~15K to ~8K tokens per forward pass
- **Attention-Guided Compression**: Preserves spatially coherent voxel groupings
- **Fixed-Size Outputs**: Q-Former ensures consistent memory usage across subjects

### Computational Efficiency
- **Progressive Compression**: Multiple ToMer layers gradually reduce computational load
- **Shared Architecture**: No subject-specific parameters reduce total model size
- **Mixed Precision**: FP16 training with automatic loss scaling
- **Gradient Checkpointing**: Reduces memory during backpropagation

## HPC Usage

### Environment Setup
```bash
# Interactive A100 session
srun -t 4:00:00 --mem=64000 --gres=gpu:a100:1 --pty /bin/bash

# Load modules
module purge
module load cuda/11.6 cudnn/8.4.0
```

### Singularity Container
```bash
singularity exec --nv \
    --overlay /scratch/overlay.ext3:ro \
    /scratch/containers/pytorch_22.08.sif \
    /bin/bash -c "source /ext3/env.sh && python Downstream/Train.py [args]"
```

### Training Time Estimates
- **Multi-subject pretraining**: ~8-12 hours on A100 40GB
- **Single-subject evaluation**: ~30 minutes per subject
- **Memory requirements**: ~32GB GPU memory for full model

## Evaluation Metrics

VoxelFormer uses standard brain decoding evaluation metrics:

- **Top-1 Retrieval Accuracy**: Percentage of correctly retrieved images from 300-image pool
- **Forward Retrieval**: Using brain embeddings to find matching images  
- **Backward Retrieval**: Using image embeddings to find matching brain responses
- **Cosine Similarity**: Alignment between brain-derived and CLIP embeddings
- **Parameter Count**: Total trainable parameters for efficiency comparison

**Chance Level**: 0.33% (1/300 images)

## FAQ

### How does VoxelFormer achieve parameter efficiency?

VoxelFormer reduces parameters through several architectural innovations:
- **Token Merging**: Reduces input sequence length by merging redundant voxels
- **Query-Based Attention**: Fixed set of learnable queries vs full attention matrices
- **Shared Architecture**: No subject-specific layers unlike MindEye2
- **Progressive Compression**: Gradual dimension reduction instead of large MLPs

### What is the role of Token Merging in brain data?

Token Merging (ToMer) dynamically identifies and merges highly correlated voxels:
- **Spatial Coherence**: Tends to merge spatially neighboring voxels
- **Information Preservation**: Maintains task-relevant neural patterns
- **Computational Savings**: Reduces tokens from ~15K to ~8K per subject
- **Training Integration**: Applied during both training and inference

### How does multi-subject training work without subject-specific layers?

VoxelFormer achieves cross-subject generalization through:
- **Coordinate-Based Positional Encoding**: SIREN embeddings capture spatial relationships
- **Query-Driven Abstraction**: Learnable queries extract subject-invariant features  
- **CLIP Alignment**: Common visual-semantic space across subjects
- **Progressive Compression**: Hierarchical feature extraction reduces subject-specific noise

### What are the limitations compared to subject-specific methods?

Current limitations include:
- **Lower Peak Accuracy**: 74% vs 98% for state-of-the-art subject-specific models
- **Training Subject Dependency**: Performance drops for completely unseen subjects
- **Limited Reconstruction**: Focus on retrieval rather than image generation
- **Anatomical Alignment**: May benefit from additional spatial normalization

## Citation

If you use VoxelFormer in your research, please cite our paper:

```bibtex
@article{le2024voxelformer,
  title={VoxelFormer: Parameter-Efficient Multi-Subject Visual Decoding from fMRI},
  author={Le, Chenqian and Zhao, Yilin and Emami, Nikasadat and Yadav, Kushagra and Liu, Xujin Chris and Chen, Xupeng and Wang, Yao},
  journal={IEEE Conference Proceedings},
  year={2024},
  organization={New York University Tandon School of Engineering}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built upon the **Natural Scenes Dataset (NSD)** from Allen et al.
- Incorporates **CLIP** vision-language representations from OpenAI
- Inspired by **MindEye** and **MindEye2** architectures for brain decoding
- Uses **Token Merging (ToMe)** techniques for efficient transformer computation
- Thanks to the brain decoding and computational neuroscience community

## Contact

For questions about VoxelFormer, please contact:
- **Yao Wang**: yaowang@nyu.edu (Principal Investigator)
- **Chenqian Le**: cl6707@nyu.edu (Lead Author)
- **Repository**: https://github.com/kushagrayadv/voxel-former 