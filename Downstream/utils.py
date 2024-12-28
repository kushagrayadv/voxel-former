import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import random
import os
import matplotlib.pyplot as plt
import math
import webdataset as wds

import json
from PIL import Image
import requests
import time 
from accelerate import Accelerator

import logging
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        logger.info('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def batchwise_pearson_correlation(Z, B):
    # Calculate means
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # Subtract means
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # Calculate Pearson correlation coefficient
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def prenormed_batchwise_cosine_similarity(Z,B):
    return (Z @ B.T).T

def cosine_similarity(Z,B,l=0):
    Z = nn.functional.normalize(Z, p=2, dim=1)
    B = nn.functional.normalize(B, p=2, dim=1)
    # if l>0, use distribution normalization
    # https://twitter.com/YifeiZhou02/status/1716513495087472880
    Z = Z - l * torch.mean(Z,dim=0)
    B = B - l * torch.mean(B,dim=0)
    cosine_similarity = (Z @ B.T).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def get_non_diagonals(a):
    a = torch.triu(a,diagonal=1)+torch.tril(a,diagonal=-1)
    # make diagonals -1
    a=a.fill_diagonal_(-1)
    return a

def gather_features(image_features, voxel_features, accelerator):  
    all_image_features = accelerator.gather(image_features.contiguous())
    if voxel_features is not None:
        all_voxel_features = accelerator.gather(voxel_features.contiguous())
        return all_image_features, all_voxel_features
    return all_image_features

# def soft_clip_loss(preds, targs, temp=0.125):
#     clip_clip = (targs @ targs.T)/temp
#     brain_clip = (preds @ targs.T)/temp
#     loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
#     loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
#     loss = (loss1 + loss2)/2
#     return loss

def soft_clip_loss(preds, targs, accelerator=None, temp=0.125):
    # Gather tensors across all GPUs if using distributed training
    if accelerator is not None and accelerator.num_processes > 1:
        preds = torch.cat(torch.distributed.nn.all_gather(preds), dim=0)
        targs = torch.cat(torch.distributed.nn.all_gather(targs), dim=0)
    
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    # Compute the global loss
    loss = (loss1 + loss2) / 2
    
    # Scale the loss to avoid over-counting gradients.
    # By default, each GPU now sees the entire global batch and calls backward.
    # To correct for this, divide by the number of processes.    
    return loss

def soft_siglip_loss(preds, targs, temp, bias):
    temp = torch.exp(temp)
    
    logits = (preds @ targs.T) * temp + bias
    # diagonals (aka paired samples) should be >0 and off-diagonals <0
    labels = (targs @ targs.T) - 1 + (torch.eye(len(targs)).to(targs.dtype).to(targs.device))

    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels[:len(preds)])) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels[:,:len(preds)])) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco_hard_siglip_loss(preds, targs, temp, bias, perm, betas):
    temp = torch.exp(temp)
    
    probs = torch.diag(betas)
    probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

    logits = (preds @ targs.T) * temp + bias
    labels = probs * 2 - 1
    #labels = torch.eye(len(targs)).to(targs.dtype).to(targs.device) * 2 - 1
    
    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels)) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels)) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

MINFLOAT=torch.finfo(torch.bfloat16).min
MAXFLOAT=torch.finfo(torch.bfloat16).max

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator: Accelerator =None, local_rank=None, bidirectional=True):

    if accelerator is not None and accelerator.num_processes > 1:
        preds = torch.cat(torch.distributed.nn.all_gather(preds), dim=0)
        targs = torch.cat(torch.distributed.nn.all_gather(targs), dim=0)
        if betas is not None:
            betas = accelerator.gather(betas)
        if perm is not None:
            perm = accelerator.gather(perm.to(preds.device)) # perm is not cuda
    

    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2

    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
    
    return loss
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))
    return trainable
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def resize(img, img_size=128):
    if img.ndim == 3: img = img[None]
    return nn.functional.interpolate(img, size=(img_size, img_size), mode='nearest')

pixcorr_preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])
def pixcorr(images,brains,nan=True):
    all_images_flattened = pixcorr_preprocess(images).reshape(len(images), -1)
    all_brain_recons_flattened = pixcorr_preprocess(brains).view(len(brains), -1)
    if nan:
        corrmean = torch.nanmean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    else:
        corrmean = torch.mean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    return corrmean

def select_annotations(annots, random=True):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[j] != '':
                    t = b[j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt

from generative_models.sgm.util import append_dims
def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples

#  Numpy Utility 
def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        
# Torch fwRF
def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125):
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp

    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def save_ckpt(tag, args, model, diffusion_prior, optimizer, lr_scheduler, epoch, losses, test_losses, lrs, accelerator, ckpt_saving=True):
    # TODO: refactor according to new configuration system
    outdir = os.path.abspath(f'../train_logs/{args.model_name}')
    if not os.path.exists(outdir) and ckpt_saving:
        os.makedirs(outdir,exist_ok=True)
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        save_dict = {
            'epoch': epoch,
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
        }
        if diffusion_prior is not None:
            save_dict['diffusion_prior'] = accelerator.unwrap_model(diffusion_prior).state_dict()
    
        torch.save(save_dict, ckpt_path)
    logger.info(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(args, model, diffusion_prior=None, optimizer=None, lr_scheduler=None, accelerator=None, tag='last', strict=True):
    """
    Load checkpoint for model, optimizer, and training state.
    If specified tag not found, will try to find latest iteration checkpoint.
    
    Args:
        args: Training arguments
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        lr_scheduler: Optional learning rate scheduler to load state
        accelerator: Accelerator instance for distributed training
        tag: Checkpoint tag to load (default: 'last')
        strict: Whether to strictly enforce that the keys in state_dict match
        
    Returns:
        epoch: Last epoch number
        losses: Training loss history
        test_losses: Validation loss history 
        lrs: Learning rate history
        resumed: Whether checkpoint was successfully loaded
    """
    resumed = False
    epoch = 0
    losses = []
    test_losses = []
    lrs = []
    
    # Construct checkpoint path
    # TODO: Add support for loading from specific checkpoint path
    ckpt_dir = os.path.abspath(f'../train_logs/{args.model_name}')
    ckpt_path = os.path.join(ckpt_dir, f'{tag}.pth')
    
    # If specified checkpoint doesn't exist, try to find latest iteration checkpoint
    if not os.path.exists(ckpt_path):
        logger.info(f"Checkpoint {ckpt_path} not found, searching for latest iteration checkpoint...")
        if os.path.exists(ckpt_dir):
            # Find all iteration checkpoints
            iter_ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('iter_') and f.endswith('.pth')]
            if iter_ckpts:
                # Extract iteration numbers and find latest
                iter_nums = [int(f.split('_')[1].split('.')[0]) for f in iter_ckpts]
                latest_iter = max(iter_nums)
                ckpt_path = os.path.join(ckpt_dir, f'iter_{latest_iter}.pth')
                logger.info(f"Found latest iteration checkpoint: {ckpt_path}")
            else:
                logger.info("No iteration checkpoints found")
                return epoch, losses, test_losses, lrs, resumed
    
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        # Load checkpoint on CPU to avoid GPU RAM spike
        ckpt = torch.load(ckpt_path, map_location='cpu')
        diffusion_prior_state_dict = None
        if diffusion_prior is not None and 'diffusion_prior' in ckpt:
            diffusion_prior_state_dict = ckpt['diffusion_prior']
        elif diffusion_prior is not None:
            # raise ValueError("Diffusion prior state not found in checkpoint, but have diffusion prior model")
            logger.info("Diffusion prior state not found in checkpoint, but have diffusion prior model")
        elif diffusion_prior is None and 'diffusion_prior' in ckpt:
            raise ValueError("Diffusion prior state found in checkpoint, but no diffusion prior model")
        else:
            logger.info("No diffusion prior model")

        # Load model state
        if accelerator is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(ckpt['model_state_dict'], strict=strict)
            if diffusion_prior is not None and diffusion_prior_state_dict is not None:
                unwrapped_diffusion_prior = accelerator.unwrap_model(diffusion_prior)
                unwrapped_diffusion_prior.load_state_dict(diffusion_prior_state_dict, strict=strict)               
        else:
            model.load_state_dict(ckpt['model_state_dict'], strict=strict)
            if diffusion_prior is not None and diffusion_prior_state_dict is not None:
                diffusion_prior.load_state_dict(diffusion_prior_state_dict, strict=strict)
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if lr_scheduler is not None and 'lr_scheduler' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        
        # Load training state
        epoch = ckpt.get('epoch', 0)
        losses = ckpt.get('train_losses', [])
        test_losses = ckpt.get('test_losses', [])
        lrs = ckpt.get('lrs', [])
        
        logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
        resumed = True
        
        # Handle wandb logging resume
        if args.wandb_log and accelerator is not None and accelerator.is_main_process:
            import wandb
            if wandb.run is not None:
                # Log metrics from previous training
                for i, (loss, test_loss, lr) in enumerate(zip(losses, test_losses, lrs)):
                    wandb.log({
                        "train/loss": loss,
                        "test/loss": test_loss,
                        "train/lr": lr,
                        "train/epoch": i
                    })
        
    else:
        logger.info(f"No checkpoint found at {ckpt_path}, starting from scratch")
    
    return epoch, losses, test_losses, lrs, resumed