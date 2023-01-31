import torch
from torchvision.models import resnet50


class ResNetEncoder(torch.nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return self.head(x)


class ResNetHead(torch.nn.Module):
    def __init__(self, fc):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        aux = x
        x = self.fc(x)
        return x, aux


def load_moco_v2(remove_last_downsampling=False):
    model = resnet50()
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 128, bias=False),
    )

    params = torch.load('misc/moco_v2_800ep_pretrain.pth.tar')['state_dict']
    prefix = 'module.encoder_q.'
    model.load_state_dict({k: params[prefix + k] for k in model.state_dict()})

    head = ResNetHead(model.fc)
    model.fc = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()

    if remove_last_downsampling:
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)

    return ResNetEncoder(model, head)
