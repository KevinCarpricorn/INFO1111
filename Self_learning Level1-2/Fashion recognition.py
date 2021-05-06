import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# training set
train_set = torchvision.datasets.FashionMNIST(
    root='D:\\OneDrive - The University of Sydney (Students)\\INFO1111\\Self_learning Level1-2\\Fashion_MNIST\\.data.FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)

# testing set
test_set = torchvision.datasets.MNIST(
    root='D:\\OneDrive - The University of Sydney (Students)\\INFO1111\\Self_learning Level1-2\\Fashion_MNIST\\.data.FashionMNIST',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)


# building network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t


# training model
network = Network()

optimizer = optim.Adagrad(network.parameters(), lr=0.01)

for epoch in range(20):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(f'epoch: {epoch + 1}, total correct: {total_correct}, loss: {total_loss}, Accuracy: {total_correct / len(train_set)}')
