import os
import json
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms
import scipy
import math

def get_transforms(image_size):
    '''
    Returns image transforms.
    '''
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    tx = {}
    
    tx['train'] = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx


class ds_multilabel(Dataset):
    def __init__(self, phase, args, tx):
        self.dataset_name = args.dataset
        self.num_classes = args.num_classes
        self.path_to_dataset = args.path_to_dataset

        self.image_paths = np.load(os.path.join(self.path_to_dataset, f'formatted_{phase}_images.npy'))#image_ids
        self.label_matrix = np.load(os.path.join(self.path_to_dataset, f'formatted_{phase}_labels.npy'))#label_matrix

        self.tx = tx[phase]

        
    def inject_noise(self, args):
        
        clip_logits = np.load(os.path.join(args.path_to_dataset, 'clip_logits.npy'))
        pos_logits = clip_logits[self.label_matrix == 1]
        neg_logits = clip_logits[self.label_matrix == 0]

        subtractive_threshold = np.percentile(pos_logits, args.noise_rate)
        additive_threshold = np.percentile(pos_logits, 100 - args.noise_rate)
        

        num_subtractive = 0 
        num_additive = 0 

        for i in range(self.label_matrix.shape[0]):
            for j in range(self.label_matrix.shape[1]):
                if self.label_matrix[i][j] == 0 and clip_logits[i][j] > additive_threshold:
                    self.label_matrix[i][j] = 1 # additive noise
                    num_additive += 1
                elif self.label_matrix[i][j] == 1 and clip_logits[i][j] < subtractive_threshold:
                    self.label_matrix[i][j] = 0 # subtractive noise
                    num_subtractive += 1
        print(f'Additive noise : {num_additive} {np.sum(neg_logits > additive_threshold)}')
        print(f'Subtractive noise : {num_subtractive} {np.sum(pos_logits < subtractive_threshold)}')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        with Image.open(image_path) as I_raw:
            I = self.tx(I_raw)
        label = torch.FloatTensor(np.copy(self.label_matrix[idx, :]))
        return I, label, idx

def get_dataset(args):
    tx = get_transforms(args.image_size)
    dataset_train = ds_multilabel('train', args, tx)
    dataset_test = ds_multilabel('test', args, tx)
    return {'train': dataset_train, 'test': dataset_test}