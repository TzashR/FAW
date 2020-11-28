import torch
import Data_Loader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class FAW_Net(Module):
    def __Init__(self):
        super(FAW_Net, self).__init__()

#TODO figure out how many output channels (filters?),stride, padding
        self.cnn1 =  Conv2d(in_channels=4,out_channels=128, kernel_size=5,stride = 2),



        self.cnn2 = Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=2),
        self.dropout = Dropout(0.5)
        self.fc1 = Linear(128,32)
        self.fc2= Linear(32,8)
        self.fc3 = Linear(8,1)

        def forward(self, x):
            x = self.cnn1(x)
            ReLU(x)
            MaxPool2d(x)
            self.cnn2(x)
            ReLU(x)
            MaxPool2d(x)
            self.fc1(x)
            ReLU(x)
            BatchNorm2d(x)
            self.dropout(x)
            self.fc2(x)
            ReLU(x)
            BatchNorm2d(x)
            self.dropout(x)
            self.fc3(x)
            return Softmax(x)


