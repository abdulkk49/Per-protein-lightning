import random, torch
import os, numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from os.path import join, exists, dirname, abspath, realpath

pwd = dirname(realpath("__file__"))
loc_dict = {'Cell.membrane': 0, 'Cytoplasm': 1, 'Endoplasmic.reticulum': 2, 'Extracellular': 3, 'Golgi.apparatus': 4, 'Lysosome/Vacuole': 5, 'Mitochondrion': 6, 'Nucleus': 7, 'Peroxisome': 8, 'Plastid': 9}
mem_dict = {'M': 0, 'S': 1, 'U': 2}

class PerProteinDataset(Dataset):
    def __init__(self, loclabelprefix, memlabelprefix, input_ids, attention_mask):
        self.loclabelprefix = loclabelprefix
        self.memlabelprefix = memlabelprefix
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.loclabels = np.loadtxt(self.loclabelprefix)
        self.memlabels = np.loadtxt(self.memlabelprefix)
        self.num_sample = self.loclabels.shape[0]
    
    def __getitem__(self, index):
       
        at_m = self.attention_mask[index]
        in_id = self.input_ids[index]
        loclabel = torch.Tensor(np.array(self.loclabels[index]))
        memlabel = torch.Tensor(np.array(self.memlabels[index]))
       
        return in_id, at_m, loclabel, memlabel

    def __len__(self):
        return self.num_sample
