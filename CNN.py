import torch
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, Sequential, Flatten,BatchNorm1d
import torch.nn.functional as F


class PrintLayer(Module):
    '''
    A layer that prints the current x. Used for debugging inside the NN
    '''
    def __init__(self, layer_name):
        super(PrintLayer, self).__init__()
        self.layer_name = layer_name

    def forward(self, x):
        print(f' layer {self.layer_name} {x.shape}')
        return x


class FawNet(Module):
    def __init__(self):
        super(FawNet, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)
        self.pool1 = MaxPool2d(kernel_size=5)
        self.conv2 = Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.pool2 = MaxPool2d(kernel_size=5)
        self.fc1 = Linear(16*63*37,16)
#        self.dense1_bn = BatchNorm1d(32)
        self.dropout1 = Dropout(0.5)
        self.fc2 = Linear(16, 8)
        self.dropout2 = Dropout(0.5)
        self.fc3 = Linear(8, 1)

    def forward(self, x):
#        print('@@@1 ', x.shape)
        x = self.conv1(x)
        F.relu(x, inplace=True)
        x = self.pool1(x)
#        print('@@@2 ', x.shape)
        x = self.conv2(x)
        F.relu(x, inplace = True)
        x = self.pool2(x)
#        print('@@@3 ', x.shape)
        x = torch.flatten(x,1)
#        print('@@@4 ', x.shape)
        x = self.fc1(x)
        F.relu(x,inplace=True)
#        x = self.dense1_bn(x)
#        print('@@@5 ', x.shape)
        x = self.dropout1(x)
        x = self.fc2(x)
        F.relu(x, inplace=True)
#        print('@@@6 ', x.shape)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

        # self.cnn_layers = Sequential(
        #     Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=3),
        #     Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=5),
        #     Flatten(),
        #     Linear(106848,32),
        #     ReLU(inplace=True),
        #     BatchNorm1d(32),
        #     Dropout(),
        #     Linear(32, 32),
        #     ReLU(),
        #     BatchNorm1d(32),
        #     Dropout(),
        #     Linear(32, 1),
        # )


    # def forward(self, x):
    #     x = self.cnn_layers(x)
    #     return x

