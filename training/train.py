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
import shutil
import torch.nn.functional as F

## functions
def train(epoch,pp):
    net.train(True)
    train_loss = 0
    dataloader = train_dataloader
    for batch_idx, (x_data,fb_data,e_data,onset_data,l_data,t_data) in enumerate(dataloader):
        x_data = x_data.to(device)
        fb_data = fb_data.to(device)
        e_data = e_data.to(device)
        onset_data = onset_data.to(device)
        l_data = l_data.to(device)
        t_data = t_data.to(device)
        optimizer.zero_grad()
        out = net(x_data,onset_data)
        loss = lossF(out,fb_data,e_data,l_data,epoch,pp,t_data)
        
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx, len(dataloader),100. * batch_idx / len(dataloader), loss.item()))
    net.train(False)
    with torch.no_grad():
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader)))
        history['Train Loss'].append(train_loss / len(dataloader))
        # if epoch >= int(EPOCHS*3/4) - 1:
        torch.save(net.state_dict(),'./model/net'+str(epoch)+'.pt')

def test(dataloader,model):
    model.eval()
    data_pred = []
    with torch.no_grad():
        for (x_data,fb_data,e_data,onset_data,l_data,t_data) in dataloader:
            x_data = x_data.to(device)
            onset_data = onset_data.to(device)
            out = model(x_data,onset_data)

            data_pred.extend(out.cpu().detach().numpy().tolist())
            
    return np.array(data_pred)

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        ft_data = sio.loadmat('./data/ft.mat')['ft']
        ft_data = torch.from_numpy(ft_data).type(torch.complex64)
        self.ft_data = ft_data.to(device)

        hrirs_all = sio.loadmat('./data/train_data.mat')['hrirs_all']
        hrirs_front = hrirs_all[55::126,:,:]
        front_max = np.max(hrirs_front,axis=2)

        self.front_max = torch.from_numpy(front_max).type(torch.float32).to(device)
        
    def forward(self, hrir_pred, ori_fb, ori_p, labels, epoch, pp,t_data):

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
        
        # 计算每个类别的中心
        centers = torch.zeros((num_classes, pred_mag.shape[1],pred_mag.shape[2])).to(hrir_pred.device)
        for i, label in enumerate(unique_labels):
            centers[i,:,:] = torch.mean(pred_mag[torch.squeeze(labels == label),:,:], dim=0)

        # 计算类间距离
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

        
        # 总损失
        loss_cls = inter_distances# - intra_distances/100
        # print(loss_cls)


        loss = loss_p + loss_fb*1 + loss_cls*1 + F.mse_loss(pred_mag,torch.cat((t_data,torch.flip(t_data[:,:,1:128],dims=[2])),dim=2))*1
        
        if epoch == EPOCHS-1:
            print(loss_p)
            print(loss_fb)
            print(loss_cls)

        if pp == 55:
            (front_pred_max,_) = torch.max(hrir_pred,dim=2)
            loss_front = F.mse_loss(self.front_max,front_pred_max)
            loss = loss + loss_front*100


        return loss

seed = 660
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_ids = [0]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 80
EPOCHS = 1024

## LOAD ORI DATA
hrirs_all = sio.loadmat('./data/train_data.mat')['hrirs_all']
hrtfs_all = sio.loadmat('./data/train_data.mat')['hrtfs_all']
labels_all = sio.loadmat('./data/train_data.mat')['labels_all']
onsets_all = sio.loadmat('./data/train_data.mat')['onsets_all']
target_fb = sio.loadmat('./data/target_dtf.mat')['target_dtf']
target_e = sio.loadmat('./data/target_e.mat')['target_e']
print(hrirs_all.shape)
print(hrtfs_all.shape)
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


for pp in range(126):
    print('#################'+str(pp)+'###############')


    train_hrirs_fake = hrirs_fake[pp::126,:,:]
    train_target_fb  = target_fb[pp::126,:,:]
    train_target_e   = target_e[pp::126,:]
    train_onseth_all = onseth_all[pp::126,:,:]
    train_labels_all = labels_all[pp::126,:]
    train_hrtfs_all  = hrtfs_all[pp::126,:,:]

    print('Load Train Data:')
    train_dataset = myDataset(train_hrirs_fake,train_target_fb,train_target_e,train_onseth_all,train_labels_all,train_hrtfs_all)
    train_dataloader = DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = False)

    ## Initial NET
    net = hrir_net()
    if len(device_ids) > 1:
        net = nn.DataParallel(net,device_ids=device_ids)
    net = net.to(device)
    lossF = My_loss()#nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma = 0.9)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    history = {'Train Loss':[],'Valid Loss':[]}

    ## Train
    torch.cuda.empty_cache()
    for epoch in range(EPOCHS):
        train(epoch,pp)
        # scheduler.step()

    ## plot loss
    net.train(False)
    plt.clf()
    plt.plot(history['Train Loss'][5:],label = 'Train Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig('./loss/Loss_'+str(pp)+'.png')
    print(min(history['Train Loss']))
    best_model_ind = history['Train Loss'].index(min(history['Train Loss']))
    #.index(min(history['Valid Loss'][-int(EPOCHS/4):]))
    print(best_model_ind)
    print(history['Train Loss'][best_model_ind])

    ## 仅保存最好的模型，节省硬盘空间
    best_model_state_dict = torch.load('./model/net'+str(best_model_ind)+'.pt')
    shutil.rmtree('./model')
    os.mkdir('./model')
    torch.save(best_model_state_dict,'./model/best_model_'+str(pp)+'.pt')

    ## Initial NET
    net = hrir_net()
    if len(device_ids) > 1:
        net = nn.DataParallel(net,device_ids=device_ids)
    net = net.to(device)
    net.load_state_dict(best_model_state_dict)
    net.eval()

    # evalute train set
    hrir_pred = test(train_dataloader,net)
    print(hrir_pred.shape)
    sio.savemat('./pred_hrir_pp/hrir_pred_'+str(pp)+'.mat', {'hrir_pred':hrir_pred})