import numpy as np
import torch
import matplotlib.pyplot as plt
import shap
import os

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

def main():
    print("Running Main Script")
    nfiles = 10
    batch_size = 64
    num_epochs = 5
    device = torch.device("cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.Dropout(),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(320, 50),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(50, 10),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(-1, 320)
            x = self.fc_layers(x)
            return x


    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output.log(), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output.log(), target).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist_data", train=False, transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    batch = next(iter(test_loader))
    images, _ = batch

    background = images[:100]

    assert os.path.exists('example_pictures')
    image_files = os.listdir('example_pictures')

    for i in range (nfiles):
        test_images = images[i*5:i*5+5]

        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(test_images)

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        filename = f'example_pictures/shap_image_{i}.png'
        shap.image_plot(shap_numpy, -test_numpy, show = False)
        plt.savefig(filename)

    print("Script finished")

if __name__ == '__main__':
    main()