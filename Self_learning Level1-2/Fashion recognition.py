import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from plot import plot_confusion_matrix
import time


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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))

        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=150)
        self.out = nn.Linear(in_features=150, out_features=10)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv3(t)
        t = F.relu(t)
        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 32 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.dropout(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        t = self.dropout(t)

        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t


# training model
network = Network()

# He initialization
for m in network.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')

optimizer = optim.Adam(network.parameters(), lr=0.1)

start = time.time()
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
end = time.time()
print(f"time: {end - start}")


# testing "plotting a confusion matrix"
@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


test_loader = DataLoader(test_set, batch_size=100)
test_preds = get_all_preds(network, test_loader)
cm = confusion_matrix(test_set.targets, test_preds.argmax(dim=1))
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, test_set.classes)
plt.show()
