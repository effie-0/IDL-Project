import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from classifier import ResidualBlock, ResNet18
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCHSIZE=256

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

def train(model, train_loader, optimizer, criterion,  epochs):
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        total = 0.0
        train_acc = 0.0
        train_loss = []
        total_labels = []

        model.train()
        for batch_num, (feats, labels) in enumerate(train_loader):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(feats)
                loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            prob = F.softmax(outputs.detach(), dim=1)

            _, pred_labels = torch.max(prob, 1)
            pred_labels = pred_labels.view(-1)
            train_acc += torch.sum(torch.eq(pred_labels, labels)).item()    
            
            total += len(labels)
            train_loss.extend([loss.item()]*feats.size()[0])
            torch.cuda.empty_cache()

            del loss
            del feats
            del labels
            del pred_labels
            del outputs
            del prob

        scheduler.step()

        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    #'scheduler_state_dict' : scheduler.state_dict(),
        }, "Model_"+str(epoch))
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print('epoch: %d\t'%(epoch+1),  'time: %d m: %d s '% divmod(time.time() - start_time, 60))
        start_time = time.time()
        print('train_loss: %.5f\ttrain_acc: %.5f' %(np.mean(train_loss), train_acc/total))
        print('val_loss: %.5f\tval_acc: %.5f'% (val_loss, val_acc))
        print('*'*60)