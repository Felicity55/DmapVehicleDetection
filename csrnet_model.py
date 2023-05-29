import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.models.vgg import make_layers, cfgs, vgg16_bn
from ptflops import get_model_complexity_info

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=1):
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        
            in_channels = v
        # print("sigmoid")
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_cfg = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_cfg, batch_norm=True)
        self.backend = make_layers(self.backend_cfg, in_channels=512, batch_norm=True, dilation=2)
        # print(self.backend.size)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        if not load_weights:
            mod = vgg16_bn(pretrained=True)
            self._initialize_weight()
            od = OrderedDict()
            
            for front_key in self.frontend.state_dict():
                vgg_key = 'features.' + front_key
                od[front_key] = mod.state_dict()[vgg_key]

            self.frontend.load_state_dict(od)


    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        # print(x.shape)
        # x=x.flatten()
        # print(x.shape)
        # x=nn.Sigmoid(x)
        x = self.output_layer(x)
        # print(x.shape)
        
        x = F.interpolate(x, scale_factor=8, mode='bilinear')
        return x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# testing
if __name__ == '__main__':
    csrnet = CSRNet()
    input_tensor = torch.ones((1, 3, 512, 512))
    outputs = csrnet(input_tensor)
    print(outputs.shape)

    macs, _ = get_model_complexity_info(csrnet, (3, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
    print(macs)
