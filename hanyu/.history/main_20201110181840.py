import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from classifier 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(100),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0, 360),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(100),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    ])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=data_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def evaluate(model, data_loader, criterion):
    avg_loss = 0.0
    total = 0.
    accuracy = 0.
    test_loss = []
    model.eval()
    for batch_num, (feats, labels) in enumerate(data_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
        prob = F.softmax(outputs.detach(), dim=1)
        
        _, pred_labels = torch.max(prob, 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])

        torch.cuda.empty_cache()
        del outputs
        del feats
        del labels
        del pred_labels
        del prob

    return np.mean(test_loss), accuracy/total

criterion = nn.CrossEntropyLoss()

model = ResNet18(ResidualBlock).to(device)

# model = DenseNet121().to(device)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)