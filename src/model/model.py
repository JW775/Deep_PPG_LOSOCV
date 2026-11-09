import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPPG(nn.Module):
    def __init__(self):
        super(DeepPPG, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(7,3), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxPool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_N1 = nn.Conv2d(16, 32, kernel_size=(1,3), stride=1, padding=0)
        self.bn_N1 = nn.BatchNorm2d(32)
        self.maxPool_N1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_N2 = nn.Conv2d(32, 64, kernel_size=(1,3), stride=1, padding=0)
        self.bn_N2 = nn.BatchNorm2d(64)
        self.maxPool_N2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_N3 = nn.Conv2d(64, 128, kernel_size=(1,3), stride=1, padding=0)
        self.bn_N3 = nn.BatchNorm2d(128)
        self.maxPool_N3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_N4 = nn.Conv2d(128, 256, kernel_size=(1,3), stride=1, padding=0)
        self.bn_N4 = nn.BatchNorm2d(256)
        self.maxPool_N4 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_N5 = nn.Conv2d(256, 512, kernel_size=(1,3), stride=1, padding=0)
        self.bn_N5 = nn.BatchNorm2d(512)
        self.maxPool_N5 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_N6 = nn.Conv2d(512, 1024, kernel_size=(1,3), stride=1, padding=0)
        self.bn_N6 = nn.BatchNorm2d(1024)
        self.maxPool_N6 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv_N7 = nn.Conv2d(1024, 2048, kernel_size=(1,3), stride=1, padding=0)
        self.bn_N7 = nn.BatchNorm2d(2048)
        self.maxPool_N7 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(2048, 32, kernel_size=(1,1), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (B, 4, 7, 1025)
    
    # ---- 前置层 ----
        x = self.conv1(x)              # (B, 8, 7, 1025)
        x = self.bn1(x)
        x = F.elu(x)
    
        x = self.conv2(x)              # (B, 16, 1, 1023)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.maxPool1(x)           # (B, 16, 1, 511)
    
        # ---- 循环层 N_L=7 ----
        x = self.conv_N1(x)            # (B, 32, 1, 509)
        x = self.bn_N1(x)
        x = F.elu(x)
        x = self.maxPool_N1(x)         # (B, 32, 1, 254)
        
        x = self.conv_N2(x)            # (B, 64, 1, 252)
        x = self.bn_N2(x)
        x = F.elu(x)
        x = self.maxPool_N2(x)         # (B, 64, 1, 126)
        
        x = self.conv_N3(x)            # (B, 128, 1, 124)
        x = self.bn_N3(x)
        x = F.elu(x)
        x = self.maxPool_N3(x)         # (B, 128, 1, 62)
        
        x = self.conv_N4(x)            # (B, 256, 1, 60)
        x = self.bn_N4(x)
        x = F.elu(x)
        x = self.maxPool_N4(x)         # (B, 256, 1, 30)
        
        x = self.conv_N5(x)            # (B, 512, 1, 28)
        x = self.bn_N5(x)
        x = F.elu(x)
        x = self.maxPool_N5(x)         # (B, 512, 1, 14)
        
        x = self.conv_N6(x)            # (B, 1024, 1, 12)
        x = self.bn_N6(x)
        x = F.elu(x)
        x = self.maxPool_N6(x)         # (B, 1024, 1, 6)
        
        x = self.conv_N7(x)            # (B, 2048, 1, 4)
        x = self.bn_N7(x)
        x = F.elu(x)
        x = self.maxPool_N7(x)         # (B, 2048, 1, 2)
        
        # ---- 最后 1×1 卷积 ----
        x = self.conv3(x)              # (B, 32, 1, 2)
        x = self.bn3(x)
        x = F.elu(x)
        
        # ---- 展平 ----
        x = self.flatten(x)            # (B, 64)
        
        # ---- 全连接 ----
        x = self.fc1(x)                # (B, 512)
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)                # (B, 1)
        
        return x

if __name__ == '__main__':
    model = DeepPPG()
    dummy = torch.randn(4, 4, 7, 1025)   # batch=4
    out = model(dummy)
    print(out.shape)    # torch.Size([4, 1]) 
            