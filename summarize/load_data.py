from torchvision import datasets, transforms
from torch.utils import data
import torch
from torch.utils.data import Dataset

class AdvDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def get_data(return_classes = False, verbose = False):
    '''
    get CIFAR-10 dataset
    
    return: trainset, trainloader, testset, testloader, (classes)
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, 
                                download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=4, 
                                shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, 
                            download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=1, 
                                shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if verbose:
        print("train set length = ", len(trainset))
        print("test set length = ", len(testset))
    if return_classes:
        return trainset, trainloader, testset, testloader, classes
    else:
        return trainset, trainloader, testset, testloader