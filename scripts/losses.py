import torch
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F
import numpy as np

class My_loss(nn.Module):
    """
	This is a class for building loss functions 
	"""
    def __init__(self,device,config):
        super().__init__()
        ft_data = sio.loadmat(config.ft_mat_pth)['ft']
        ft_data = torch.from_numpy(ft_data).type(torch.complex64)
        self.ft_data = ft_data.to(device)

        hrirs_all = sio.loadmat(config.dataset_mat_pth)['hrirs_all']
        hrirs_front = hrirs_all[55::126,:,:]
        front_max = np.max(hrirs_front,axis=2)

        self.front_max = torch.from_numpy(front_max).type(torch.float32).to(device)
        self.cls_loss_w = config.cls_loss_w
        
    def forward(self, hrir_pred, ori_fb, ori_p, labels, pp, t_data):

        ### rms
        pred_p = torch.sqrt(torch.mean(hrir_pred**2,dim=2))
        loss_p = F.mse_loss(ori_p,pred_p)

        
        ### fb
        pred_fb = torch.zeros_like(ori_fb).to(ori_fb.device)#Bx2x28
        FFT_L = 2400*3 - 1
        hrir_pred_cc = torch.zeros((hrir_pred.shape[0],hrir_pred.shape[1],FFT_L)).to(hrir_pred.device)
        hrir_pred_cc[:,:,:256] = hrir_pred
        pred_fft = torch.fft.fft(hrir_pred_cc,FFT_L,dim=2)

        temp_f = pred_fft[:,:,:,None]*self.ft_data
        temp_h = torch.real(torch.fft.ifft(temp_f,FFT_L,dim=2)).type(torch.float32)
        temp_h = 2*temp_h[:,:,:2400,:]
        temp_h = torch.sqrt(F.relu(temp_h))
        temp_h = torch.sqrt(torch.mean(temp_h**2,dim=2))
        pred_fb = temp_h


        loss_fb = F.mse_loss(ori_fb,pred_fb)


        ### cls
        unique_labels = torch.unique(labels)
        num_classes = unique_labels.shape[0]

        pred_mag = torch.abs(torch.fft.fft(hrir_pred,256,dim=2))
        
        # calculate the center of each class
        centers = torch.zeros((num_classes, pred_mag.shape[1],pred_mag.shape[2])).to(hrir_pred.device)
        for i, label in enumerate(unique_labels):
            centers[i,:,:] = torch.mean(pred_mag[torch.squeeze(labels == label),:,:], dim=0)

        # inter_distances
        inter_distances = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                inter_distances += torch.norm(centers[i,:,:] - centers[j,:,:])
        inter_distances /= (num_classes * (num_classes - 1) / 2)

        # # 计算类内距离
        # intra_distances = 0
        # for i, label in enumerate(labels):
        #     intra_distances += torch.norm(pred_mag[i,:,:] - centers[torch.squeeze(unique_labels == label),:,:])
        # intra_distances /= labels.shape[0]

        
        loss_cls = inter_distances# - intra_distances/100
        # print(loss_cls)


        loss = loss_p + loss_fb*1 + loss_cls*self.cls_loss_w + F.mse_loss(pred_mag,torch.cat((t_data,torch.flip(t_data[:,:,1:128],dims=[2])),dim=2))*1
        #cls: 0 0.01 0.1 1

        if pp == 55:
            (front_pred_max,_) = torch.max(hrir_pred,dim=2)
            loss_front = F.mse_loss(self.front_max,front_pred_max)
            loss = loss + loss_front*100


        return loss