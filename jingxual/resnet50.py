def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    
# kernel_size=1, padding=0
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False)

num_classes = 10
class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, isDownSample=False):
        super(Bottleneck, self).__init__()
        
        self.inchannel = inchannel
        self.expansion = 4
        self.isDownSample = isDownSample
        
        self.conv1 = conv1x1(inchannel, outchannel)
        self.norm1 = nn.BatchNorm2d(outchannel)

        self.conv2 = conv3x3(outchannel, outchannel, stride)
        self.norm2 = nn.BatchNorm2d(outchannel)
        
        self.conv3 = conv1x1(outchannel, outchannel * self.expansion)
        self.norm3 = nn.BatchNorm2d(outchannel * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        if isDownSample:
            self.downsample = nn.Sequential(
                conv1x1(inchannel, outchannel * self.expansion, stride),
                nn.BatchNorm2d(outchannel * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.relu(self.norm3(self.conv3(out)))

        if self.isDownSample:
            out += self.downsample(identity)
        
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=num_classes, outs=[64, 128, 256, 512]):
        super(ResNet50, self).__init__()
        """
        Arguments:
            block (class): BasicBlock(nn.Module)
            layers (list): A ResNet’s layer is composed of the same blocks stacked one after the other.
            num_classes (int): num_classes = 4000
            outs (list): dim before expension(*4)
        """
        self.expansion = 4
        self.inchannel = 64*self.expansion
        self.conv0 = conv3x3(3, 64*self.expansion, stride=1)
        
        self.layer1=self.make_layer(block,outs[0],layers[0],stride=1) # 3
        self.layer2=self.make_layer(block,outs[1],layers[1],stride=2) # 4
        self.layer3=self.make_layer(block,outs[2],layers[2],stride=2) # 6
        self.layer4=self.make_layer(block,outs[3],layers[3],stride=2) # 3

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
        
        # self.cfc = nn.Linear(512*4, outFeat)
        # self.crelu = nn.ReLU(inplace=True)

    def make_layer(self, block, out_channels, block_num, stride=1):
        """
            block (class): BottleneckBlock(nn.Module)
            out_channels (int)：output size of layer
            block_num (int)：total blocks
            stride (int)：Conv Block stride
        """

        if stride!=1 or self.inchannel!=(out_channels*self.expansion):
            isDownsample = True
        else: isDownsample = False
            
        layers = []
        #Conv Block: different size
        conv_block=block(self.inchannel, out_channels, stride, isDownsample)
        layers.append(conv_block)
        self.inchannel = out_channels*self.expansion
        
        #Identity Block: same size
        for i in range(1, block_num):
            layers.append(block(self.inchannel, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, ver=False):
        out = x
        out = self.conv0(out)

        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)

        out = self.avgpool(out)
        # out = torch.squeeze(out)
        out = out.reshape(out.shape[0], out.shape[1])
        
        # embed = out
        out = self.fc(out)
        # cout = self.cfc(out)
        # cout = self.crelu(cout)
        return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)
        
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
