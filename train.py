import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random
import time
import matplotlib.pyplot as plt

from model import Refiner, Detector

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Random Seed
seed = random.randint(1, 10000)
torch.manual_seed(seed)

# Parameters
batch_size = 1
nEpochs = 5
lr = 0.0001

# Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_train = MNIST(root='MNIST_data', train=True, transform=transform, download=True)
train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# Model
refiner = Refiner().to(device)
detector = Detector().to(device)


# Initialize weights
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


refiner.apply(initialize_weights)
detector.apply(initialize_weights)

# Loss function and Optimizers
criterion_BCE = nn.BCELoss()
criterion_MSE = nn.MSELoss()
optimizerG = optim.Adam(refiner.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.9)
optimizerD = optim.Adam(detector.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.9)

# Train
D_losses = []
R_losses = []


def train(epoch):
    iter = 0
    for batch, (images, labels) in enumerate(train_dataloader):
        iter += 1
        real_images, labels = images.to(device), labels.to(device)
        fake_images = refiner(real_images)

        # Discriminator: maximize log(D(x)) + log(1-D(R(X)))
        detector.zero_grad()
        real_results = detector(real_images.detach())
        real_targets = torch.ones_like(real_results, device=device)

        fake_results = detector(fake_images.detach())
        fake_targets = torch.zeros_like(fake_results, device=device)
        errorD_real = criterion_BCE(real_results, real_targets)
        errorD_fake = criterion_BCE(fake_results, fake_targets)
        errorD = errorD_real + errorD_fake
        errorD.backward()
        optimizerD.step()

        P_real = real_results.mean().item()
        P_fake = fake_results.mean().item()  # discriminator가 진짜라고 판별할 확률

        # Generator: maximize log(D(G(z)))
        refiner.zero_grad()

        fake_images = refiner(real_images)
        fake_results = detector(fake_images)
        fake_labels = torch.ones_like(fake_results, device=device)
        errorR_BCE = criterion_BCE(fake_results, fake_labels)
        errorR_MSE = criterion_MSE(real_images, fake_images)

        if errorR_MSE < 0.005:
            return

        errorR = errorR_MSE*0.4 + errorR_BCE
        errorR.backward()
        optimizerG.step()

        if iter % 20 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, nEpochs, batch + 1, len(train_dataloader),
                     errorD.item(), errorR.item()))

        R_losses.append(errorR.item())
        D_losses.append(errorD.item())


start = time.time()
errorR_MSE = 0
for epoch in range(1, nEpochs+1):
    train(epoch)
    if errorR_MSE < 0.005:
        break

print("time: ", time.time()-start)

# Save Model
torch.save(refiner.state_dict(), 'refiner')
torch.save(detector.state_dict(), 'detector')

# Loss Visualization
plt.figure(figsize=(10,5))
plt.title("Refiner and Detector Loss During Training")
plt.plot(R_losses, label="R")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('RDloss.png')
