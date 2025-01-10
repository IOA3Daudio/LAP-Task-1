import torch
import torch.nn as nn
import numpy as np

class trainer(object):
    def __init__(self,configs,device,train_dataloader,optimizer,lossF):
        seed = configs.seed
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.device = device
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lossF = lossF
        self.model_save_pth = configs.model_save_pth
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=configs.scheduler_step_size,gamma = configs.scheduler_gamma)

    def train(self,net,pp,epoch,history):
        net.train(True)
        train_loss = 0
        dataloader = self.train_dataloader
        device = self.device
        optimizer = self.optimizer
        lossF = self.lossF
        for batch_idx, (x_data,fb_data,e_data,onset_data,l_data,t_data) in enumerate(dataloader):
            x_data = x_data.to(device)
            fb_data = fb_data.to(device)
            e_data = e_data.to(device)
            onset_data = onset_data.to(device)
            l_data = l_data.to(device)
            t_data = t_data.to(device)
            optimizer.zero_grad()
            out = net(x_data,onset_data)
            loss = lossF(out,fb_data,e_data,l_data,pp,t_data)
            
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
            torch.save(net.state_dict(), self.model_save_pth+'net'+str(epoch)+'.pt')

        # self.scheduler.step()

        return history