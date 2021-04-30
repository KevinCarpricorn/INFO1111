import torch
import torchvision
import torchvision.transforms as transforms


train_set = torchvision.datasets.MNIST(
    root='D:\OneDrive - The University of Sydney (Students)\INFO1111\.data.MNIST'
    , train=True
    , download=True
    , transform=transforms.Compose([transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(train_set)

sample = next(iter(train_set))
image, label = sample
batch = next(iter(train_loader))
images, labels = batch

