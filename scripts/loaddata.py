import scipy.io as sio
import numpy as np
from scripts.datasets import myDataset
from torch.utils.data import DataLoader



class load_origin_data(object):
    """
	This is a class for loadding dataset
    Attributes:
		target_fb: 	target features in ERBs
		target_e:   target energy
        hrirs_fake: input random samples
        onsets_all: TOAs for reconstructing HRIRs
        labels_all: labels for classification loss
        hrtfs_all:  HRTF magnitude
	"""
    def __init__(self,config):
        np.random.seed(config.seed)

        self.hrirs_all = sio.loadmat(config.dataset_mat_pth)['hrirs_all']
        self.hrtfs_all = sio.loadmat(config.dataset_mat_pth)['hrtfs_all']
        self.labels_all = sio.loadmat(config.dataset_mat_pth)['labels_all']
        self.onsets_all = sio.loadmat(config.dataset_mat_pth)['onsets_all']
        self.target_fb = sio.loadmat(config.fb_mat_pth)['target_dtf']
        self.target_e = sio.loadmat(config.e_mat_pth)['target_e']
        print(self.hrirs_all.shape)
        print(self.hrtfs_all.shape)
        print(self.target_fb.shape)
        print(self.target_e.shape) 

        self.hrirs_fake = np.random.rand(self.hrirs_all.shape[0],self.hrirs_all.shape[1],self.hrirs_all.shape[2])
        self.onseth_all = self.modify_onsets()

        self.BATCH_SIZE = config.batch_size
        
        
    def modify_onsets(self):
        ### onset to hrir shape
        onseth_all = np.zeros_like(self.hrirs_all)
        for i in range(onseth_all.shape[0]):
            for j in range(onseth_all.shape[1]):
                temp_onset = self.onsets_all[i,j]
                onseth_all[i,j,temp_onset:] = 1
                onseth_all[i,j,-100:] = 0

        return onseth_all
    
    def gen_train_dataloader(self,pp,shuffle_flag):
        train_hrirs_fake = self.hrirs_fake[pp::126,:,:]
        train_target_fb  = self.target_fb[pp::126,:,:]
        train_target_e   = self.target_e[pp::126,:]
        train_onseth_all = self.onseth_all[pp::126,:,:]
        train_labels_all = self.labels_all[pp::126,:]
        train_hrtfs_all  = self.hrtfs_all[pp::126,:,:]

        train_dataset = myDataset(train_hrirs_fake,train_target_fb,train_target_e,train_onseth_all,train_labels_all,train_hrtfs_all)
        train_dataloader = DataLoader(dataset = train_dataset,batch_size = self.BATCH_SIZE,shuffle = shuffle_flag)
        return train_dataloader
    
        