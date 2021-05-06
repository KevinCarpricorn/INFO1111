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


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# training set
train_set = torchvision.datasets.MNIST(
    root='D:\\OneDrive - The University of Sydney (Students)\\INFO1111\\Self_learning Level1-2\\.data.MNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)

# testing set
test_set = torchvision.datasets.MNIST(
    root='D:\\OneDrive - The University of Sydney (Students)\\INFO1111\\Self_learning Level1-2\\.data.MNIST',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2, 2)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2, 2)
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.out(t)
        return t


network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)


# training
def train(epoch):
    for epoch in range(epoch):
        total_loss = 0
        total_correct = 0
        # train a epoch
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


train(8)
# testing "plotting a confusion matrix"
@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


test_loader = DataLoader(test_set, batch_size=1000)
test_preds = get_all_preds(network, test_loader)
cm = confusion_matrix(test_set.targets, test_preds.argmax(dim=1))
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, test_set.classes)
plt.show()
