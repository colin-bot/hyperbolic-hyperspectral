import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18
from torchvision.models import resnet34

from models_hypll import PoincareResNetModel

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(204,10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 5, 5)
        self.fc1 = nn.Linear(10080, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.flatten(x) # for MSELoss
        return x


class ClassificationNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(204, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 5, 5)
        self.conv2_bn = nn.BatchNorm2d(5)
        self.fc1 = nn.Linear(10080, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, n_classes) # binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, n_classes=2):
        super(ResNet, self).__init__()
        self.inchannel = 10
        self.conv1 = nn.Sequential(
            nn.Conv2d(204, 10, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 10, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, n_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FCNet(nn.Module):
    def __init__(self, n_classes=2):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(204, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, n_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x     


class HSCNN(nn.Module): #https://github.com/cogsys-tuebingen/deephs_fruit/blob/master/classification/models/deephs_net.py
    def __init__(self, args, num_classes=3, base_dim=204, hidden_layers=[25, 30, 50]):
        super(HSCNN, self).__init__()
        bands = base_dim//args.pooling_factor
        self.bands = bands
        kernel_count = 3
        assert len(hidden_layers) == 3

        self.conv = nn.Sequential(
            nn.Conv2d(bands, bands * kernel_count, kernel_size=7, padding=1, groups=bands),
            nn.Conv2d(bands * kernel_count, hidden_layers[0], kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(hidden_layers[0]),
            nn.Conv2d(hidden_layers[0], hidden_layers[0] * kernel_count, kernel_size=3, padding=1, groups=hidden_layers[0]),
            nn.Conv2d(hidden_layers[0] * kernel_count, hidden_layers[1], kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(hidden_layers[1]),
            nn.Conv2d(hidden_layers[1], hidden_layers[1] * kernel_count, kernel_size=3, padding=1, groups=hidden_layers[1]),
            nn.Conv2d(hidden_layers[1] * kernel_count, hidden_layers[2], kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_layers[2]),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.Linear(hidden_layers[2], num_classes),
        )

        self.init_params()

    def init_params(self):
        '''Init layer parameters.'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, _x, channel_wavelengths=None):
        out = self.conv(_x)
        out = out.view(_x.shape[0], -1)
        out = self.fc(out)
        return out


def get_resnet(args, n_classes, base_dim):
    model = resnet34()

    if args.onebyoneconv:
        model.conv1 = nn.Sequential(
            nn.Conv2d(base_dim//args.pooling_factor, args.onebyoneconvdim, kernel_size=1),
            nn.BatchNorm2d(args.onebyoneconvdim),
            nn.ReLU(),
            nn.Conv2d(args.onebyoneconvdim, 64, kernel_size=(7,7), stride=(3,3), padding=(3,3), bias=False)
        )
    else:
        model.conv1 = nn.Conv2d(base_dim//args.pooling_factor, 64, kernel_size=(7, 7), stride=(3,3), padding=(3,3), bias=False)
    if not args.classification: n_classes = 1
    if args.combined_loss: n_classes += 1
    # model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
    model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
    return model


def get_base_dim(args):
    base_dim=204 # for the WUR kiwi HSI dataset
    if args.dataset_label_type == 'cifar': base_dim = 3 # for Cifar10 (RGB)
    elif args.dataset_label_type == 'deephs': base_dim = 224 # for DeepHS_Fruit HSI dataset
    return base_dim


def get_output_dim(args, n_classes):
    output_dim = n_classes
    if args.combined_loss: output_dim += 1
    elif not args.classification: output_dim = 1
    return output_dim


def get_model(args, n_classes=2):
    output_dim = get_output_dim(args, n_classes)
    base_dim = get_base_dim(args)

    if args.special_modes:
        if 'avg1d' in args.special_modes.split('-'):
            model = FCNet(n_classes=output_dim)
    elif args.hypll:

        model = PoincareResNetModel(args,
                                    n_classes=output_dim,
                                    base_dim=base_dim,
                                    channel_sizes=[4, 8, 16],
                                    group_depths=[5, 4, 3],
                                    manifold_type='poincare')
    else:
        if args.resnet:
            model = get_resnet(args, output_dim, base_dim)
        else:
            if args.classification:
                # model = ClassificationNet(n_classes=output_dim)
                model = HSCNN(args, num_classes=output_dim, base_dim=base_dim)
            else:
                model = RegressionNet(n_classes)
    return model
