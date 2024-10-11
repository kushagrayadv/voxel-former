import nibabel as nib
import torch
import numpy as np
import random
import h5py
import re
import webdataset as wds
from torch.utils.data import Dataset, DataLoader

from IPython import embed

class MindEye2Dataset(Dataset):
    def __init__(self, args, data_type, split='train'):
        self.dataset, subj_list = load_web_dataset(args, split)
        self.voxels, self.num_voxels = load_voxels(args, subj_list, data_type)
        self.images = load_images(args)
        self.samples = list(iter(self.dataset))
        self.coords = {}
        for subj in subj_list:
            # f = h5py.File(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj0{subj}/func1pt8mm/roi', 'r')
            mask =  nib.load(f'/scratch/cl6707/Shared_Datasets/NSD/nsddata/ppdata/subj0{subj}/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata() == 1
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            coords = torch.nonzero(mask, as_tuple=False).float().to(device)
            self.coords[f'subj0{subj}'] = coords
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        behav0, past_behav0, future_behav0, old_behav0, subj_id = self.samples[idx]
        image_id = int(behav0[0,0])
        voxel_id = int(behav0[0,5])
        subj_id = int(subj_id)
        coord = self.coords[f'subj0{subj_id}']

        return self.images[image_id], self.voxels[f'subj0{subj_id}'][voxel_id].view(1,-1), subj_id, coord

def my_split_by_node(urls): return urls

subject_pattern = re.compile(r"/subj0(\d+)/")

def add_subject(sample):
    match = subject_pattern.search(sample["__url__"])
    if match:
        sample["subject_id"] = int(match.group(1))
    return sample

def load_web_dataset(args, split):
    if args.multi_subject:
        subj_list = np.arange(1,9)
        subj_list = subj_list[subj_list != args.subj]
        nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
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

def pad_collate_fn(batch):
    images, voxels, subj_idx = zip(*batch)
    max_length = max(item.shape[1] for item in voxels)

    padded_voxels = [
        torch.nn.functional.pad(item, (0, max_length - item.shape[1]))
        for item in voxels
    ]
    padded_voxels = torch.stack(padded_voxels, dim=0)

    return images, padded_voxels, subj_idx


# if __name__ == "__main__":
#     from Train import parse_arguments
#     args = parse_arguments()
#     data_type = torch.float16

#     train_data = MindEye2Dataset(args, data_type, 'train')
#     test_data = MindEye2Dataset(args, data_type, 'test')

#     train_dl = DataLoader(
#         train_data, 
#         batch_size=32,
#         shuffle=True,
#         num_workers=4,
#         collate_fn=pad_collate_fn
#     )

#     test_dl = DataLoader(
#         test_data, 
#         batch_size=32,
#         shuffle=True,
#         num_workers=4,
#         collate_fn=pad_collate_fn
#     )