import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

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
batch_size = 16

# Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_test = MNIST(root='MNIST_data', train=False, transform=transform, download=True)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

fashion_mnist_test = FashionMNIST(root='Fashion_MNIST_data', train=False, transform=transform, download=True)
fashion_dataloader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=True)

# Model
refiner = Refiner().to(device)
detector = Detector().to(device)

refiner.load_state_dict(torch.load('refiner'))
detector.load_state_dict(torch.load('detector'))

# Evaluation
real_images, real_labels = next(iter(test_dataloader))
fake_images = refiner(real_images.to(device))

real_score = detector(real_images.to(device))
fake_score = detector(fake_images)

x = [i for i in range(batch_size)]
y_real = [real_score.squeeze().cpu().detach().numpy()[i] for i in range(batch_size)]
y_fake = [fake_score.squeeze().cpu().detach().numpy()[i] for i in range(batch_size)]

# Show Image with plt
fig = plt.figure()
rows = 2
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(make_grid(real_images.cpu(), normalize=True).permute(1, 2, 0))
ax1.set_title('Real_Images_Refiner')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 3)
ax2.imshow(make_grid(fake_images.cpu(), normalize=True).permute(1, 2, 0))
ax2.set_title('Fake_Images_Refiner')
ax2.axis("off")

ax3 = fig.add_subplot(rows, cols, 2)
ax3.plot(x, y_real)
ax3.set_title('Real_Images_scores')

ax3 = fig.add_subplot(rows, cols, 4)
ax3.plot(x, y_fake)
ax3.set_title('Fake_Images_scores')

plt.tight_layout()
plt.show()
