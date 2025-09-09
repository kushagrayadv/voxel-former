import nibabel as nib
import torch
import numpy as np
import random
import h5py
import re
import os
from glob import glob
import webdataset as wds
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from tqdm import tqdm

from IPython import embed

class MindEye2Dataset(Dataset):
    def __init__(self, args, data_type, split='train', is_mni_data=False):
        self.data_type = data_type
        self.is_mni_data = is_mni_data
        self.dataset, subj_list = load_web_dataset(args, split)
        self.voxels, self.num_voxels = load_voxels(args, subj_list, data_type)
        self.images = load_images(args)
        self.samples = list(iter(self.dataset))
        self.coords = {}
        for subj in subj_list:
            mask =  nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj0{subj}/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata() == 1
            mask = torch.tensor(mask, dtype=torch.bool)
            coords = torch.nonzero(mask, as_tuple=False).float()
            self.coords[f'subj0{subj}'] = coords

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        behav0, past_behav0, future_behav0, old_behav0, subj_id = self.samples[idx]
        image_id = int(behav0[0,0])
        voxel_id = int(behav0[0,5])
        subj_id = int(subj_id)
        coord = self.coords[f'subj0{subj_id}']
        subj_key = f'subj0{subj_id}'
        
        image = torch.tensor(self.images[image_id],dtype=self.data_type)
        voxel = None
        
        # Check if the data is MNI conditioned data
        if self.is_mni_data:
            reg_data, reg_mask = load_mni_data(args, subj_key, voxel_id, self.data_type)
            
            if reg_data is not None and reg_mask is not None:
                coord = torch.nonzero(reg_mask, as_tuple=False).float()
                voxel = reg_data[reg_mask].view(1, -1)
            else:
                return image, voxel, subj_id, coord, image_id
        else:
            voxel = self.voxels[subj_key][voxel_id].view(1,-1)

        print("voxel", voxel.shape, coord.shape, reg_data.shape, reg_mask.shape)
        # Print device information for debugging
        return image, voxel, subj_id, coord, image_id

class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = []
        self.shuffle = shuffle
        self.create_batches()

    def create_batches(self):
        subject_to_indices = defaultdict(list)

        for idx, sample in enumerate(self.dataset.samples):
            subj_id = int(sample[-1])
            subject_to_indices[subj_id].append(idx)

        batches = []
        for indices in subject_to_indices.values():
            if self.shuffle:
                random.shuffle(indices)
            num_indices = len(indices) // self.batch_size * self.batch_size # drop last
            for i in range(0, num_indices, self.batch_size):
                batch = indices[i:i + self.batch_size]
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        self.create_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def my_split_by_node(urls): return urls

subject_pattern = re.compile(r"/subj0(\d+)/")

def add_subject(sample):
    match = subject_pattern.search(sample["__url__"])
    if match:
        sample["subject_id"] = int(match.group(1))
    return sample

def load_web_dataset(args, split):
    if args.multi_subject:
        subj_list = args.multi_subject
        nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
        if args.num_sessions<32:
            nsessions_allsubj =  np.array([args.num_sessions]*8)
    else:
        subj_list = [args.subj]

    if split == 'train':
        if args.multi_subject:
            url = [
                f"{args.data_path}/wds/subj0{s}/train/{i}.tar"
                for s in subj_list
                for i in range(nsessions_allsubj[s - 1])
            ]
        else:
            url = f"{args.data_path}/wds/subj0{args.subj}/train/" + "{0.." + f"{args.num_sessions - 1}" + "}.tar"

    if split == 'test':
        subj = args.subj
        if not args.new_test:  # using old test set
            num_test_mapping = {3: 2113, 4: 1985, 6: 2113, 8: 1985}
            num_test = num_test_mapping.get(subj, 2770)
            url = f"{args.data_path}/wds/subj0{subj}/test/" + "0.tar"
        else:  # using larger test set
            num_test_mapping = {3: 2371, 4: 2188, 6: 2371, 8: 2188}
            num_test = num_test_mapping.get(subj, 3000)
            url = f"{args.data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    
    dataset = (
        wds.WebDataset(url, resampled=False, nodesplitter=my_split_by_node)
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .map(add_subject)
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav", "subject_id")
    )

    return dataset, subj_list

def load_voxels(args, subj_list, data_type):
    voxels = {}
    num_voxels = {}
    for s in subj_list:
        f = h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
        betas = f['betas'][:]
        betas = torch.Tensor(betas).to("cpu").to(data_type)
        num_voxels[f'subj0{s}'] = betas[0].shape[-1]
        voxels[f'subj0{s}'] = betas
    return voxels, num_voxels

def load_images(args):
    f = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
    images = f['images']
    return images

def load_mni_data(args, subj_key, voxel_id, data_type):
    fpath = os.path.join(args.mni_data_path, f'{subj_key}', f'{subj_key}_condition_{voxel_id}_registered_to_mni.nii.gz')
    match = glob(fpath)

    if len(match) == 0:
        print(f"No MNI data found for {subj_key} {voxel_id}")
        return None
    
    try:
        nii_data = nib.load(fpath).get_fdata().astype(np.float32)
        data = torch.from_numpy(nii_data).to(data_type)
    except Exception as e:
        print(f"Error loading MNI data for {subj_key} {voxel_id}: {e}")
        return None

    mask_file_path = os.path.join(args.mni_mask_path, f'{subj_key}', f'{subj_key}_mask_registered_to_mni.nii.gz')
    match = glob(mask_file_path)
    if len(match) == 0:
        print(f"No MNI mask data found for {subj_key} {voxel_id}")
        return None
    
    mask = nib.load(mask_file_path).get_fdata().astype(np.float32)
    mask = torch.from_numpy(mask).to(torch.bool)
    
    return data, mask

def custom_collate_fn(batch):
    images, voxels, subjects, coords, image_idx = zip(*batch)
    images = torch.stack(images, dim=0)
    voxels = torch.stack(voxels, dim=0)
    subjects = torch.tensor(subjects)
    coords = torch.stack(coords, dim=0)
    image_idx = torch.tensor(image_idx)

    return images, voxels, subjects, coords, image_idx

class Args:
    def __init__(self, **kwargs):
        self.data_path = "/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2"
        self.cache_dir = "/scratch/cl6707/Shared_Datasets/NSD_MindEye/Mindeye2"
        self.mni_data_path = "/scratch/ky2684/shared-datasets/nsd-mni-dataset/"
        self.mni_mask_path = "/scratch/ky2684/shared-datasets/nsd-mni-mask/"

        # Subject and session info
        self.subj = 2
        self.num_sessions = 40

        # Multi-subject (list of subject IDs)
        self.multi_subject = [2, 3, 4, 5, 6, 7, 8]

        # Whether to use the new test set
        self.new_test = True

if __name__ == "__main__":
    args = Args()
    
    data_type = torch.float16

    batch_size = 8

    train_data = MindEye2Dataset(args, data_type, 'train', is_mni_data=True)
    sampler = SubjectBatchSampler(train_data, batch_size)
    dataloader = DataLoader(train_data, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)
    
    for i, (images, voxels, subjects, coords, image_idx) in enumerate(dataloader):
        print(images.shape, voxels.shape, subjects.shape, coords.shape, image_idx.shape)
        break
    