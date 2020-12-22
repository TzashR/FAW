# %%
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, Sequential, Flatten
from torchsummary import summary


class FawNet(Module):
    def __init__(self):
        super(FawNet, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3),
            Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=5),
            Flatten(),
            Linear(316,195040),
            ReLU(inplace=True),
            BatchNorm2d(16),
            Dropout(),
            Linear(32, 32),
            ReLU(),
            BatchNorm2d(32),
            Dropout(),
            Linear(32, 1),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        return x

