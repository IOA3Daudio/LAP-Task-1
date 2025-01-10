import torch
import numpy as np
from torch.utils.data import Dataset

class myDataset(Dataset):
    """
	This is a class for Dataset
    Attributes:
		target_fb: 	target features in ERBs
		target_e:   target energy
        hrirs_fake: input random samples
        onsets_all: TOAs for reconstructing HRIRs
        labels_all: labels for classification loss
        hrtfs_all:  HRTF magnitude
	"""
    def __init__(self,hrirs_fake, target_fb, target_e,onsets_all,labels_all,hrtfs_all):
        self.target_fb  = torch.from_numpy(target_fb).type(torch.float32)
        self.target_e   = torch.from_numpy(target_e).type(torch.float32)
        self.hrirs_fake = torch.from_numpy(hrirs_fake).type(torch.float32)
        self.onsets_all = torch.from_numpy(onsets_all).type(torch.float32)
        self.labels_all = torch.from_numpy(labels_all).type(torch.float32)
        self.hrtfs_all  = torch.from_numpy(hrtfs_all).type(torch.float32)

    
    
    def __getitem__(self, index):
        return self.hrirs_fake[index,:,:], self.target_fb[index,:,:], self.target_e[index,:], self.onsets_all[index,:,:], self.labels_all[index,:],self.hrtfs_all[index,:,:]

    def __len__(self):
        return self.hrirs_fake.shape[0]