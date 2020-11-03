import torch
import torch.nn as nn
import torch.nn.functional as F
from deform_cov import ConvOffset2D

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class stn(nn.Module):
    def __init__(self):
        super(stn, self).__init__()
        self.localization = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=5),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True),
        nn.Conv2d(8, 10, kernel_size=5),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
        nn.Linear(10 * 5 * 5, 32),
        nn.ReLU(True),
        nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class stn1(nn.Module):
    def __init__(self):
        super(stn1, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 128)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10,stn_=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.stn_=stn_
        self.net1=nn.Sequential(stn())
        self.net2=nn.Sequential(stn1())
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding = 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,stride=1, padding = 1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,stride=1, padding = 1, bias=False)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,stride=1, padding = 1, bias=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding = 1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,64, num_blocks[0], stride=1)
        self.in_planes*=2
        self.layer2 = self._make_layer(block,128, num_blocks[1], stride=1)
        self.in_planes*=2
        self.layer3 = self._make_layer(block,256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block,512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stn_==1 :
            x=self.net1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(F.relu(self.conv2(out)))
        out = self.maxpool(out)
        out = self.dropout10(out)
        out = self.layer1(out)
        if self.stn_==1 :
            out=self.net2(out)
        out = self.bn2(F.relu(self.conv3(out)))
        out = self.bn2(F.relu(self.conv4(out)))
        out = self.avgpool(out)
        out = self.dropout10(out)
        out = self.layer2(out)
        out = self.bn3(F.relu(self.conv5(out)))
        out = self.bn3(F.relu(self.conv6(out)))
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet1(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10,stn_=0):
        super(ResNet1, self).__init__()
        self.in_planes = 64
        self.stn_=stn_
        self.net1=nn.Sequential(stn())
        self.net2=nn.Sequential(stn1())
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block,128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block,256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block,512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stn_==1 :
            x=self.net1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.stn_==1 :
            out=self.net2(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(stn):
    return ResNet(BasicBlock, [2,2,2,2],stn_=stn)

def ResNet34(stn):
    return ResNet1(BasicBlock, [3,4,6,3],stn_=stn)

def ResNet50(stn):
    return ResNet(Bottleneck, [3,4,6,3],stn_=stn)

def ResNet101(stn):
    return ResNet(Bottleneck, [3,4,23,3],stn_=stn)

def ResNet152(stn):
    return ResNet(Bottleneck, [3,8,36,3],stn_=stn)

class dResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10,stn_=0):
        super(dResNet, self).__init__()
        self.in_planes = 64
        self.stn_=stn_
        self.net1=nn.Sequential(stn())
        self.net2=nn.Sequential(stn1())
        self.offset1=ConvOffset2D(3)
        self.offset2=ConvOffset2D(16)
        self.offset3=ConvOffset2D(32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block,128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block,256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block,512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stn_==1 :
            x=self.net1(x)
        x=self.offset1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out=self.offset2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out=self.offset3(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.layer1(out)
        if self.stn_==1 :
            out=self.net2(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def dResNet18(stn):
    return dResNet(BasicBlock, [2,2,2,2],stn_=stn)

def dResNet34(stn):
    return dResNet(BasicBlock, [3,4,6,3],stn_=stn)

def dResNet50(stn):
    return dResNet(Bottleneck, [3,4,6,3],stn_=stn)

def dResNet101(stn):
    return dResNet(Bottleneck, [3,4,23,3],stn_=stn)

def dResNet152(stn):
    return dResNet(Bottleneck, [3,8,36,3],stn_=stn)

