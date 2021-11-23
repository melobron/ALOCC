import torch
import torch.nn as nn


# Models for MNIST 28*28
class Refiner(nn.Module):
    def __init__(self, in_channel=1, inter_channel=64):
        super(Refiner, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel, eps=1e-06),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channel, inter_channel*2, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channel*2, inter_channel*4, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channel*4, inter_channel*8, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel*8),
            nn.LeakyReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(inter_channel*8, inter_channel*4, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel*4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(inter_channel*4, inter_channel*2, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel*2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(inter_channel*2, inter_channel, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(inter_channel, in_channel, kernel_size=5, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        noise = torch.randn_like(x)
        x = x + noise
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Detector(nn.Module):
    def __init__(self, input_size=28, in_channel=1, inter_channel=64):
        super(Detector, self).__init__()

        self.in_channel = in_channel
        self.input_size = input_size

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channel, inter_channel * 2, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channel * 2, inter_channel * 4, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel * 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channel * 4, inter_channel * 8, kernel_size=5, bias=False),
            nn.BatchNorm2d(inter_channel * 8),
            nn.LeakyReLU(inplace=True)
        )

        self.outdim = self.compute_outdim()
        self.fc = nn.Linear(self.outdim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_outdim(self):
        test_tensor = torch.Tensor(1, self.in_channel, self.input_size, self.input_size)
        for p in self.decoder.parameters():
            p.requires_grad = False
        test_tensor = self.decoder(test_tensor)
        out_dim = torch.prod(torch.tensor(test_tensor.shape[1:])).item()
        for p in self.decoder.parameters():
            p.requires_grad = True

        return out_dim

    def forward(self, x):
        x = self.decoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        out = self.sigmoid(x)
        return out
