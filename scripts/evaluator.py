import torch
import torch.nn as nn
import numpy as np

class evaluator(object):
    """
	This is a class for reconstructing HRIRs
	"""
    def __init__(self,device,dataloader):
        self.device = device
        self.dataloader = dataloader

    def test(self,net):       
        net.eval()
        data_pred = []
        with torch.no_grad():
            for (x_data,fb_data,e_data,onset_data,l_data,t_data) in self.dataloader:
                x_data = x_data.to(self.device)
                onset_data = onset_data.to(self.device)
                out = net(x_data,onset_data)

                data_pred.extend(out.cpu().detach().numpy().tolist())
                
        return np.array(data_pred)