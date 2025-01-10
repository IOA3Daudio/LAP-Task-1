from importlib import reload
import torch
import torch.nn as nn
import shutil
import os
import scipy.io as sio

from scripts.loaddata import load_origin_data
from scripts.models import hrir_net
from scripts.losses import My_loss
from scripts.trainer import trainer
from scripts.utils import plot_loss
from scripts.evaluator import evaluator

import config
reload(config)
from config import DefaultConfig

def main():

    configs = DefaultConfig()

    ori_dataset = load_origin_data(configs)

    device = torch.device("cuda:{}".format(configs.device_ids[0]) if torch.cuda.is_available() else "cpu")

    lossF = My_loss(device,configs)

    print(configs.pos_sum)

    for pp in range(configs.pos_sum):
        print('#################'+str(pp)+'###############') # temp position index

        print('Load Train Data:')
        train_dataloader = ori_dataset.gen_train_dataloader(pp,shuffle_flag = False)

        ## Initial NET
        net = hrir_net()
        if len(configs.device_ids) > 1:
            net = nn.DataParallel(net,device_ids=configs.device_ids)
        net = net.to(device)
        total = sum([param.nelement() for param in net.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))
        
        optimizer = torch.optim.Adam(net.parameters())

        mytrainer = trainer(configs,device,train_dataloader,optimizer,lossF)

        ## Train
        history = {'Train Loss':[],'Valid Loss':[]}
        torch.cuda.empty_cache()
        for epoch in range(configs.epochs):
            mytrainer.train(net,pp,epoch,history)
        
        net.train(False)

        best_model_ind = plot_loss(configs,history,pp)
        best_model_state_dict = torch.load(configs.model_save_pth+'net'+str(best_model_ind)+'.pt')
        shutil.rmtree(configs.model_save_pth)
        os.mkdir(configs.model_save_pth)
        torch.save(best_model_state_dict,configs.model_save_pth+'best_model_'+str(pp)+'.pt')

        ## Output the harmonized HRIRs
        net = hrir_net()
        if len(configs.device_ids) > 1:
            net = nn.DataParallel(net,device_ids=configs.device_ids)
        net = net.to(device)
        net.load_state_dict(best_model_state_dict)
        net.eval()

        mytest = evaluator(device,train_dataloader)
        hrir_pred = mytest.test(net)
        sio.savemat(configs.hrirs_out_pth+'hrir_pred_'+str(pp)+'.mat', {'hrir_pred':hrir_pred})

if __name__ == "__main__":
    main()
