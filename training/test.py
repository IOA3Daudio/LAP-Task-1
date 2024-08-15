import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_processing import myDataset
from build_net import hrir_net
import scipy.io as sio

## functions
def test(dataloader,model):
    model.eval()
    data_pred = []
    with torch.no_grad():
        for (x_data,fb_data,e_data,onset_data,l_data) in dataloader:
            x_data = x_data.to(device)
            onset_data = onset_data.to(device)
            out = model(x_data,onset_data)

            data_pred.extend(out.cpu().detach().numpy().tolist())
            
    return np.array(data_pred)



seed = 666
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device_ids = [2]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024

## LOAD ORI DATA
hrirs_all = sio.loadmat('./data/train_data.mat')['hrirs_all']
labels_all = sio.loadmat('./data/train_data.mat')['labels_all']
onsets_all = sio.loadmat('./data/train_data.mat')['onsets_all']
target_fb = sio.loadmat('./data/target_dtf.mat')['target_dtf']
target_e = sio.loadmat('./data/target_e.mat')['target_e']
print(hrirs_all.shape)
print(target_fb.shape) 
print(target_e.shape) 

hrirs_fake = np.random.rand(hrirs_all.shape[0],hrirs_all.shape[1],hrirs_all.shape[2])

### onset to hrir shape
onseth_all = np.zeros_like(hrirs_all)
for i in range(onseth_all.shape[0]):
    for j in range(onseth_all.shape[1]):
        temp_onset = onsets_all[i,j]
        onseth_all[i,j,temp_onset:] = 1
        onseth_all[i,j,-100:] = 0
###

print('Load Train Data:')
train_dataset = myDataset(hrirs_fake,target_fb,target_e,onseth_all,labels_all)
train_dataloader = DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = False)

## Initial NET
net = hrir_net()
if len(device_ids) > 1:
    net = nn.DataParallel(net,device_ids=device_ids)
net = net.to(device)
best_model_pth = 'model/best_model.pt'
state_dict = torch.load(best_model_pth)
net.load_state_dict(state_dict)
net.eval()

# evalute train set
hrir_pred = test(train_dataloader,net)
print(hrir_pred.shape)
sio.savemat("hrir_pred.mat", {'hrir_pred':hrir_pred})

