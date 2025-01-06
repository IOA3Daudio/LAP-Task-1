import torch
import torch.nn as nn

class hrir_net(nn.Module):
    def __init__(self):
        super(hrir_net, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Linear(in_features=256*2, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(in_features=2048, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(in_features=2048, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(in_features=2048, out_features=256*2),
        )

    def forward(self, x, onseth):
        #(B,ear,freq)
        # out = self.block_1(x)

        out = self.block_1(x.view(x.shape[0],-1))
        out = out.view(x.shape[0],2,-1)

        out = out*onseth
        
        return out
    
    
    
    


















