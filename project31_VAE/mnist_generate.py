import torch
from torch import nn
import torchvision
from torchvision import transforms

from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import time

# hyperparameters
n_epochs = 3
learning_rate = 0.001
batch_size = 10
latent_size = 8

# dataset loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    "datasets", download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    "datasets", download=True, train=False, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encode_layers = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 3x14x14
            nn.ReLU(),
            nn.Conv2d(3, 6, 3, stride=1),
            nn.MaxPool2d(2, 2),  # 6x6x6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(216, latent_size + latent_size),
        )
        self.decode_layers = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 216),
            nn.ReLU(),
            nn.Linear(216, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encode_layers(x)
        mu = x[:, :latent_size]
        logvar = x[:, latent_size:]
        return mu, logvar

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z

    def decode(self, z):
        z = self.decode_layers(z)
        return z


def main():
    vae = VAE()
    criterion = nn.BCELoss(reduction="sum")
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    for epc in range(n_epochs):
        # train
        for imgs, lbls in tqdm(train_loader):
            mu, logvar = vae.encode(imgs)
            z = vae.sampling(mu, logvar)
            outs = vae.decode(z)

            optimizer.zero_grad()
            loss1 = criterion(outs, imgs.reshape(-1, 784))
            loss2 = -torch.sum(1 + logvar - mu * mu - torch.exp(logvar)) / 2
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

        # test
        avg_loss = 0.0
        ratio = 1 / len(test_loader)

        for imgs, lbls in tqdm(test_loader):
            mu, logvar = vae.encode(imgs)
            z = vae.sampling(mu, logvar)
            outs = vae.decode(z)

            loss1 = criterion(outs, imgs.reshape(-1, 784))
            loss2 = -torch.sum(1 + logvar - mu * mu - torch.exp(logvar)) / 2
            avg_loss += (loss1 + loss2) * ratio

        print(f"\nepoch: {epc+1}, average loss: {avg_loss}\n")
        avg_loss = 0.0

        # visualize
        fig, ax = plt.subplots(nrows=2, ncols=3)
        
        for i in range(3):
            img1 = test_dataset[random.randint(0, 9999)][0]
            ax[0][i].imshow(img1.reshape(28, 28))

            mu, logvar = vae.encode(img1.reshape(1, 1, 28, 28))
            img2 = vae.decode(vae.sampling(mu, logvar)).detach()
            ax[1][i].imshow(img2.reshape(28, 28))
        
        plt.show()


if __name__ == "__main__":
    main()
