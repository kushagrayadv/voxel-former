import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import kornia
from kornia.augmentation.container import AugmentationSequential
from brain_models.brain_transformer import BrainTransformer
# Add the path for SDXL unCLIP requirements
sys.path.append('generative_models/')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # bigG embedder

# Enable tf32 for faster computation
torch.backends.cuda.matmul.allow_tf32 = True

# Custom utility functions
import utils
from utils import save_ckpt
from dataset import MindEye2Dataset, SubjectBatchSampler, custom_collate_fn
import re
from brain_models.models import PriorNetwork, BrainDiffusionPrior
import pdb

import logging
logger = logging.getLogger(__name__)  # __name__ = name of the current module

# Add configuration for visualizing hierarchical downsampling if needed
def visualize_hierarchy(model, coords, device, wandb_log=False):
    """Visualize the hierarchical downsampling of brain coordinates"""
    # Check if this is a hierarchical perceiver model by looking for the downsample_layers attribute
    if (not hasattr(model, 'brain_decoder') or 
        not hasattr(model.brain_decoder, 'downsample_layers')):
        return None
    
    # Create a sample batch with correct dimensions
    batch_size = 1
    voxel_count = coords.shape[1]
    
    # Ensure dummy input has proper shape
    dummy_input = torch.ones(batch_size, voxel_count, 1, device=device)
    
    # Create a simplified visualization without running the actual downsampling
    num_levels = len(model.brain_decoder.downsample_layers) + 1
    fig = plt.figure(figsize=(15, 5))
    
    # Just visualize the original coordinates
    ax = fig.add_subplot(1, num_levels, 1, projection='3d')
    orig_coords = coords[0].cpu().numpy()
    ax.scatter(orig_coords[:, 0], orig_coords[:, 1], orig_coords[:, 2], s=5, alpha=0.8)
    ax.set_title(f'Original: {orig_coords.shape[0]} tokens')
    
    # Estimate the downsampled coordinates based on downsample factors
    current_count = voxel_count
    for i in range(num_levels-1):
        factor = model.brain_decoder.downsample_layers[i].factor
        current_count = current_count // factor
        
        # Simply sample a subset of points to visualize
        indices = torch.randperm(orig_coords.shape[0])[:current_count]
        downsampled = orig_coords[indices]
        
        ax = fig.add_subplot(1, num_levels, i+2, projection='3d')
        ax.scatter(downsampled[:, 0], downsampled[:, 1], downsampled[:, 2], s=5, alpha=0.8)
        ax.set_title(f'Level {i+1}: ~{current_count} tokens')
    
    plt.tight_layout()
    
    if wandb_log:
        return wandb.Image(fig, caption="Estimated Brain Token Downsampling")
    else:
        return fig

def prepare_data(args, data_type):
    train_data = MindEye2Dataset(args.data, data_type, 'train')
    train_sampler = SubjectBatchSampler(train_data, args.train.batch_size)
    train_dl = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True)

    test_data = MindEye2Dataset(args.data, data_type, 'test')
    test_sampler = SubjectBatchSampler(test_data, args.train.batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_sampler=test_sampler, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True)

    num_iterations_per_epoch = len(train_data) // args.train.batch_size
    return train_dl, test_dl, len(test_data), num_iterations_per_epoch

def build_model(args, device, data_type):
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=False if args.model.use_avg_pool else True,
        only_tokens=False if args.model.use_avg_pool else True,
    ).to(device)
    logger.info("clip_img_embedder")
    utils.count_params(clip_img_embedder)


    if args.train.blurry_recon:
        from diffusers import AutoencoderKL
        autoenc = AutoencoderKL(
            down_block_types=['DownEncoderBlock2D'] * 4,
            up_block_types=['UpDecoderBlock2D'] * 4,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            sample_size=256,
        ).to(device)
        ckpt = torch.load(f'{args.data.cache_dir}/sd_image_var_autoenc.pth', map_location=device)
        autoenc.load_state_dict(ckpt)
        autoenc.eval()
        autoenc.requires_grad_(False)

        from autoencoder.convnext import ConvnextXL
        cnx = ConvnextXL(f'{args.data.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth').to(device)
        cnx.requires_grad_(False)
        cnx.eval()

        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor([0.228, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

        blur_augs = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
            data_keys=["input"],
        ).to(device)
    else:
        autoenc = None
        cnx = None
        mean = None
        std = None
        blur_augs = None

    model = BrainTransformer(args).to(device)

    # Optional Prior Network
    if args.train.use_prior:
        out_dim = args.model.clip_emb_dim
        depth = 6
        dim_head = 52
        heads = args.model.clip_emb_dim // 52
        timesteps = 100
        
        prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=args.model.clip_seq_dim,
            learned_query_mode="pos_emb"
        )
        
        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )
    else:
        diffusion_prior = None
    

    # model = torch.compile(model)
    logger.info("model parameters:")
    Model_param = utils.count_params(model)

    if model.brain_encoder:
        logger.info("model.brain_encoder")
        Brain_Encoder_param = utils.count_params(model.brain_encoder)
    else:
        Brain_Encoder_param = None
    
    if model.brain_decoder:
        logger.info("model.brain_decoder")
        Brain_Decoder_param = utils.count_params(model.brain_decoder)
    else:
        Brain_Decoder_param = None
    param_count_dict = {"Model_param": Model_param, "Brain_Encoder_param": Brain_Encoder_param, "Brain_Decoder_param": Brain_Decoder_param}
    
    # Visualize hierarchical structure if using perceiver decoder
    if model.decoder_type == "perceiver" and args.wandb_log and args.model.visualize_hierarchy:
        # Create a dummy batch for visualization
        dummy_coords = torch.randn(1, 10000, 3, device=device)  # Adjust size based on your data
        hierarchy_viz = visualize_hierarchy(model, dummy_coords, device, wandb_log=True)
        if hierarchy_viz is not None:
            param_count_dict["hierarchy_visualization"] = hierarchy_viz
    
    return (
        clip_img_embedder,
        model,
        diffusion_prior,
        autoenc,
        cnx,
        mean,
        std,
        blur_augs,
        param_count_dict
    )

def setup_optimizer(args, model, diffusion_prior, num_iterations_per_epoch):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    max_lr = args.train.max_lr

    # Initialize parameter groups list
    opt_grouped_parameters = []

    # Group parameters based on decoder type
    if model.decoder_type == 'perceiver':
        # For Perceiver, only optimize decoder parameters
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.brain_decoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.brain_decoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])
    else:  # 'qformer'
        # For Q-former, optimize both encoder and decoder
        # Encoder parameters
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.brain_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.brain_encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])
        
        # Feature mapper parameters
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.feature_mapper.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.feature_mapper.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])
        
        # Decoder parameters
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.brain_decoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.brain_decoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ])

    # Add prior network parameters if enabled
    if args.train.use_prior and diffusion_prior is not None:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ])

    # Initialize optimizer
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

    # Setup learning rate scheduler
    if args.train.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.train.num_epochs * num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif args.train.lr_scheduler_type == 'cycle':
        total_steps = int(np.floor(args.train.num_epochs * num_iterations_per_epoch))
        logger.info(f"total_steps {total_steps}")
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, 
            pct_start=2 / args.train.num_epochs
        )
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler


def setup_wandb(args,train_url="", test_url=""):
    local_rank = os.getenv('RANK', 0)
    wandb_log = args.wandb_log

    if int(local_rank) == 0 and wandb_log:
        import wandb
        logger.info(f"wandb {args.wandb_project} run {args.model_name}")
        wandb.init(
            entity=args.wandb_entity,
            id=args.instance_dir,
            project=args.wandb_project,
            name=args.model_name,
            config=vars(args),
            resume="auto",
        )
    else:
        wandb_log = False
    return wandb_log


def train(args: DictConfig, model, diffusion_prior, train_dl, test_dl, accelerator, data_type, num_iterations_per_epoch,
          num_test, subj_list, clip_img_embedder, optimizer, lr_scheduler, wandb_log, autoenc, cnx, mean, std,
          blur_augs, epoch_start=0, losses=None, test_losses=None, lrs=None):
    
    device = accelerator.device
    num_epochs = args.train.num_epochs
    batch_size = args.train.batch_size
    ckpt_interval = args.train.ckpt_interval
    ckpt_saving = args.train.ckpt_saving
    mixup_pct = args.train.mixup_pct
    blur_scale = args.train.blur_scale
    clip_scale = args.train.clip_scale
    prior_scale = args.train.prior_scale
    use_image_aug = args.train.use_image_aug
    blurry_recon = args.train.blurry_recon
    use_prior = args.train.use_prior
    model_name = args.model_name

    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)

    logger.info(f"{model_name} starting with epoch {epoch_start} / {num_epochs}")
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    # Initialize tracking lists if not provided
    losses = losses if losses is not None else []
    test_losses = test_losses if test_losses is not None else []
    lrs = lrs if lrs is not None else []

    # Training loop
    epoch_progress = tqdm(
        range(epoch_start, num_epochs), 
        disable=not accelerator.is_local_main_process
    )
    
    # Create visualizations for hierarchical perceiver if needed
    if (wandb_log and accelerator.is_main_process):
        # Unwrap the model to access its attributes
        unwrapped_model = accelerator.unwrap_model(model)
        
        if (unwrapped_model.decoder_type == "perceiver" and 
            getattr(args.model, 'visualize_hierarchy', False)):
            
            # Get first batch for visualization
            for images, voxels, subj_idx, coords, image_idx in train_dl:
                coords = coords.to(device)
                hierarchy_viz = visualize_hierarchy(
                    unwrapped_model,  # Use unwrapped model here
                    coords, 
                    device, 
                    wandb_log=True
                )
                if hierarchy_viz is not None:
                    wandb.log({"hierarchy_visualization": hierarchy_viz})
                break
    
    global_iteration = epoch_start * num_iterations_per_epoch
    for epoch in epoch_progress:
        model.train()
        iteration = 0
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.
        
        recon_cossim = 0.
        test_recon_cossim = 0.
        recon_mse = 0.
        test_recon_mse = 0.

        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.
        
        loss_prior_total = 0.
        test_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1
        
        iter_progress = tqdm(train_dl, desc=f'Epoch {epoch}', leave=False, disable=not accelerator.is_local_main_process)
        for train_i, (images, voxels, subj_idx, coords, image_idx) in enumerate(iter_progress):
            blurry_pixcorr_per_iter = 0
            recon_cossim_per_iter = 0
            recon_mse_per_iter = 0
            loss_prior_per_iter = 0
            loss_clip_per_iter = 0
            loss_blurry_per_iter = 0
            loss_blurry_cont_per_iter = 0
            with torch.amp.autocast('cuda'):
                batch_size = voxels.shape[0]
                if batch_size != args.train.batch_size:
                    logger.info(f"Warning: Batch size mismatch. Expected {args.train.batch_size}, got {batch_size}")
                    continue
                optimizer.zero_grad()
                loss=0.

                lens = torch.ones(voxels.shape[0], dtype=torch.long)*voxels.shape[-1]

                # image_idx = image_idx.cpu().long().numpy()
                # _, img_sorted_idx = np.unique(image_idx, return_index=True) # this breaks multi gpu training
                # voxel0 = voxels[img_sorted_idx]
                # image = images[img_sorted_idx]
                # coords = coords[img_sorted_idx]
                voxel0 = voxels
                image = images
                coords = coords
                

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                
                if use_image_aug: 
                    image = img_augment(image)

                clip_target = clip_img_embedder(image)
                assert not torch.any(torch.isnan(clip_target))
                backbone, clip_voxels, blurry_image_enc_ = model(voxel0, coords)
                    
                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if use_prior:
                    loss_prior, prior_out = diffusion_prior(text_embed=backbone, image_embed=clip_target)

                    loss_prior_per_iter = loss_prior.item()
                    loss_prior_total += loss_prior_per_iter
                    loss_prior *= prior_scale
                    loss += loss_prior

                    recon_cossim_per_iter = nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_cossim += recon_cossim_per_iter
                    recon_mse_per_iter = mse(prior_out, clip_target).item()
                    recon_mse += recon_mse_per_iter

                if clip_scale>0:
                    if epoch < int(mixup_pct * num_epochs):                
                        loss_clip = utils.mixco_nce(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.1,
                            accelerator=accelerator,
                            perm=perm, betas=betas, select=select)
                    else:
                        epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            accelerator,
                            temp=epoch_temp)

                    loss_clip_per_iter = loss_clip.item()
                    loss_clip_total += loss_clip_per_iter
                    loss_clip *= clip_scale
                    loss += loss_clip

                if blurry_recon:     
                    image_enc_pred, transformer_feats = blurry_image_enc_

                    image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_per_iter = loss_blurry.item()
                    loss_blurry_total += loss_blurry_per_iter

                    if epoch < int(mixup_pct * num_epochs):
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                        image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                            image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                    image_norm = ((image - mean)/std)
                    image_aug = (blur_augs(image - mean))/std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)
                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        temp=0.2)
                    loss_blurry_cont_per_iter = cont_loss.item()
                    loss_blurry_cont_total += loss_blurry_cont_per_iter

                    loss += (loss_blurry + 0.1*cont_loss) * blur_scale #/.18215
                        
                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                if blurry_recon:
                    with torch.no_grad():
                        # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                        samp_size = len(image)//5 if len(image)>5 else len(image)
                        random_samps = np.random.choice(np.arange(len(image)), size=samp_size, replace=False)
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        blurry_pixcorr_per_iter = pixcorr.item()
                        blurry_pixcorr += blurry_pixcorr_per_iter
                
                try:
                    utils.check_loss(loss)
                except:
                    save_ckpt(f'nan_loss_ckpt',
                                  args,
                                  accelerator.unwrap_model(model),
                                  None if diffusion_prior is None else accelerator.unwrap_model(diffusion_prior),
                                  optimizer,
                                  lr_scheduler,
                                  epoch,
                                  losses,
                                  test_losses,
                                  lrs,
                                  accelerator,
                                  ckpt_saving=True)
                    # pdb.set_trace()
                    raise ValueError
                                
                accelerator.backward(loss)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if args.train.lr_scheduler_type is not None:
                    lr_scheduler.step()

                if accelerator.is_main_process and wandb_log:
                    wandb.log({
                        "train/loss_per_iter": loss.item(),
                        "train/blurry_pixcorr_per_iter": blurry_pixcorr_per_iter,
                        "train/recon_cossim_per_iter": recon_cossim_per_iter,
                        "train/recon_mse_per_iter": recon_mse_per_iter,
                        "train/loss_prior_per_iter": loss_prior_per_iter,
                        "train/loss_clip_per_iter": loss_clip_per_iter,
                        "train/loss_blurry_per_iter": loss_blurry_per_iter,
                        "train/loss_blurry_cont_per_iter": loss_blurry_cont_per_iter,
                    }, step=global_iteration)

                iteration += 1
                global_iteration += 1
                if accelerator.is_main_process:
                    if global_iteration % args.train.ckpt_iter == 0 and ckpt_saving:
                        save_ckpt(f'iter_{global_iteration}',
                                  args,
                                  accelerator.unwrap_model(model),
                                  None if diffusion_prior is None else accelerator.unwrap_model(diffusion_prior),
                                  optimizer,
                                  lr_scheduler,
                                  epoch,
                                  losses,
                                  test_losses,
                                  lrs,
                                  accelerator,
                                  ckpt_saving=True)
        model.eval()
        test_image, test_voxel, test_coords, test_lens = None, None, None, None
        
        # if accelerator.is_main_process:
        with torch.no_grad(), torch.amp.autocast('cuda'): 
            # Add progress bar for test dataloader
            test_progress = tqdm(test_dl, desc=f'Testing epoch {epoch}', leave=False, 
                                disable=not accelerator.is_local_main_process)
            for test_i, (images, voxels, subj_idx, coords, image_idx) in enumerate(test_progress):
                images = images.to(device)
                voxels = voxels.to(device)
                coords = coords.to(device)
                image_idx = image_idx.to(device)
                # all test samples should be loaded per batch such that test_i should never exceed 0
                if len(images) != args.train.batch_size:
                    logger.info(f"Warning: Batch size mismatch. Expected {args.train.batch_size}, got {len(images)}")
                    continue

                # Update progress bar description with current metrics
                if test_i > 0:  # Only update if we have accumulated some metrics
                    test_progress.set_postfix({
                        'loss': f"{np.mean(test_losses[-(test_i+1):]):.4f}",
                        'fwd_acc': f"{test_fwd_percent_correct/(test_i+1):.4f}",
                        'bwd_acc': f"{test_bwd_percent_correct/(test_i+1):.4f}"
                    })

                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels
                    image = image_idx
                    unique_image, sort_indices = torch.unique(image, return_inverse=True) # this will break multi gpu inference if wanting to do all clip
                    for im in unique_image:
                        locs = torch.where(im == image_idx)[0]
                        if len(locs)==1:
                            locs = locs.repeat(3)
                        elif len(locs)==2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs)==3
                        if test_image is None:
                            test_image = torch.Tensor(images[locs,0][None])
                            test_voxel = voxels[locs][None]
                            test_coords = coords[locs][None]
                        else:
                            test_image = torch.vstack((test_image, torch.Tensor(images[locs,0][None])))
                            test_voxel = torch.vstack((test_voxel, voxels[locs][None]))
                            test_coords = torch.vstack((test_coords, coords[locs][None]))
                loss=0.
                            
                test_indices = torch.arange(len(test_voxel))
                voxel = test_voxel[test_indices]
                coords = test_coords[test_indices]
                image = test_image[test_indices]

                clip_target = clip_img_embedder(image)
                for rep in range(3):
                    backbone0, clip_voxels0, blurry_image_enc_= model(voxel[:,rep], coords[:,rep])
                    if rep==0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                clip_voxels /= 3
                backbone /= 3

                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                # for some evals, only doing a subset of the samples per batch because of computational cost
                random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                
                if use_prior:
                    loss_prior, prior_out = diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior
                    # TODO: this two line was not tested
                    test_recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target[random_samps]).mean().item()
                    test_recon_mse += mse(prior_out, clip_target[random_samps]).item()
                
                if clip_scale>0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        accelerator=accelerator,
                        temp=.006)

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * clip_scale
                    loss += loss_clip

                if blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item()

                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                
                utils.check_loss(loss)                
                test_losses.append(loss.item())

            # assert (test_i+1) == 1
            logs = {
                "epoch/epoch": epoch,
                "epoch/train_loss": np.mean(losses[-num_iterations_per_epoch:]),  # Only average losses from current epoch
                "epoch/test_loss": np.mean(test_losses[-len(test_dl):]),  # Only average losses from current test run
                "epoch/lr": lrs[-1],
                "epoch/train_fwd_acc": fwd_percent_correct / (train_i + 1),
                "epoch/train_bwd_acc": bwd_percent_correct / (train_i + 1),
                "epoch/test_fwd_acc": test_fwd_percent_correct / (test_i + 1),
                "epoch/test_bwd_acc": test_bwd_percent_correct / (test_i + 1),
            }

            if clip_scale > 0:
                logs.update({
                    "epoch/train_loss_clip": loss_clip_total / (train_i + 1),
                    "epoch/test_loss_clip": test_loss_clip_total / (test_i + 1),
                })

            if blurry_recon:
                logs.update({
                    "epoch/train_loss_blurry": loss_blurry_total / (train_i + 1),
                    "epoch/train_loss_blurry_cont": loss_blurry_cont_total / (train_i + 1),
                    "epoch/train_blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                    "epoch/test_blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                })

            if use_prior:
                logs.update({
                    "epoch/train_loss_prior": loss_prior_total / (train_i + 1),
                    "epoch/test_loss_prior": test_loss_prior_total / (test_i + 1),
                    "epoch/train_recon_cossim": recon_cossim / (train_i + 1),
                    "epoch/test_recon_cossim": test_recon_cossim / (test_i + 1),
                    "epoch/train_recon_mse": recon_mse / (train_i + 1),
                    "epoch/test_recon_mse": test_recon_mse / (test_i + 1),
                })

            # if finished training or checkpoint interval, save blurry reconstructions
            if (epoch == num_epochs-1) or (epoch % ckpt_interval == 0):
                if blurry_recon:    
                    image_enc = autoenc.encode(2*image[:4]-1).latent_dist.mode() * 0.18215
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    jj=-1
                    for j in [0,1,2,3]:
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc_pred[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')

                    if wandb_log:
                        logs["test/blur_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

            if wandb_log and accelerator.is_main_process:
                # Use end-of-epoch iteration instead of global_iteration
                epoch_step = (epoch + 1) * num_iterations_per_epoch - 1
                wandb.log(logs, step=epoch_step)
                
        # Save model checkpoint and reconstruct
        if (ckpt_saving) and (epoch % ckpt_interval == 0):
            save_ckpt(f'last',
                      args,
                      accelerator.unwrap_model(model),
                      None if diffusion_prior is None else accelerator.unwrap_model(diffusion_prior),
                      optimizer,
                      lr_scheduler,
                      epoch,
                      losses,
                      test_losses,
                      lrs,
                      accelerator,
                      ckpt_saving=True)

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()

    logger.info("\n===Finished!===\n")
    if ckpt_saving:
        save_ckpt(f'last',args,accelerator.unwrap_model(model),optimizer,lr_scheduler,epoch, losses, test_losses, lrs, accelerator, ckpt_saving=True)

@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    # Add default values for hierarchical perceiver parameters if not present
    if not hasattr(args.model, 'downsample_factors'):
        args.model.downsample_factors = [2, 2, 2, 2]
    if not hasattr(args.model, 'use_residual'):
        args.model.use_residual = True
    if not hasattr(args.model, 'downsample_method'):
        args.model.downsample_method = 'grid'
    if not hasattr(args.model, 'visualize_hierarchy'):
        args.model.visualize_hierarchy = False
    
    torch._dynamo.config.optimize_ddp=False

    # Initialize accelerator first
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        device_placement=True, 
        split_batches=False, 
        mixed_precision="fp16",
        dynamo_backend="no",
        kwargs_handlers=[kwargs]
    )

    if accelerator.is_main_process:            
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    # Set up wandb only on main process
    if accelerator.is_main_process:
        if args.wandb_log:
            import wandb
            # Try to resume wandb run if it exists
            try:
                wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=args.model_name,
                    id=args.instance_dir.replace('/', '--'),
                    resume="allow",
                    config=OmegaConf.to_container(args, resolve=True)
                )
                logger.info(f"Resumed wandb run: {wandb.run.path}")
            except wandb.errors.UsageError:
                # If run doesn't exist, start new one
                wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    name=args.model_name,
                    config=vars(args)
                )
                logger.info(f"Started new wandb run: {wandb.run.path}")
    
    utils.seed_everything(args.train.seed)
    data_type = torch.bfloat16  # Change depending on your mixed_precision

    # Setup multi-GPU training
    local_rank = int(os.getenv('RANK', 0))
    # num_devices = torch.cuda.device_count()
    # if num_devices == 0:
    #     num_devices = 1
    # args.num_devices = num_devices
    # args.train['global_batch_size'] = args.train.batch_size * accelerator.num_processes
    args.train.global_batch_size = args.train.batch_size * accelerator.num_processes
    device = accelerator.device

    # Data preparation
    train_dl, test_dl, num_test, num_iterations_per_epoch = prepare_data(args, data_type)

    # Model initialization
    clip_img_embedder, model, diffusion_prior, autoenc, cnx, mean, std, blur_augs, param_count_dict = build_model(args, device, data_type)
    if args.wandb_log and accelerator.is_main_process:
        wandb.log(param_count_dict)
    optimizer, lr_scheduler = setup_optimizer(args, model, diffusion_prior, num_iterations_per_epoch)

    # Load checkpoint if exists
    epoch_start, losses, test_losses, lrs, resumed = utils.load_ckpt(
        args=args,
        model=model,
        diffusion_prior=diffusion_prior,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        tag='last'
    )

    # Prepare for distributed training
    model, diffusion_prior, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        model, diffusion_prior, optimizer, train_dl, lr_scheduler
    )

    # Print training status
    if resumed:
        logger.info(f"Resuming training from epoch {epoch_start}")
    else:
        logger.info("Starting new training run")
        epoch_start = 0


    logger.info(f"Accelerator: {accelerator}")
    logger.info(f"args: {OmegaConf.to_yaml(args, resolve=True)}")
    # Training loop
    train(
        args=args,
        model=model,
        diffusion_prior=diffusion_prior,
        train_dl=train_dl,
        test_dl=test_dl,
        accelerator=accelerator,
        data_type=data_type,
        num_iterations_per_epoch=num_iterations_per_epoch,
        num_test=num_test,
        subj_list=[args.data.subj],
        clip_img_embedder=clip_img_embedder,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wandb_log=args.wandb_log and accelerator.is_main_process,
        autoenc=autoenc,
        cnx=cnx,
        mean=mean,
        std=std,
        blur_augs=blur_augs,
        epoch_start=epoch_start,
        losses=losses,
        test_losses=test_losses,
        lrs=lrs
    )


if __name__ == "__main__":
    main()