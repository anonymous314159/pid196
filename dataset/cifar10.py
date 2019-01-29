import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable


class CIFAR10:
    """
    A wrapper of `torchvision.datasets.CIFAR10` in PyTorch
    """
    def __init__(self, root, batch_size=128, augment=True, num_workers=4):
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        if augment:
            with torch.no_grad():
                self._transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: F.pad(
                                                    Variable(x.unsqueeze(0), requires_grad=False),
                                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
                self._transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                        ])
        else:
            self._transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self._transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=self._transform_train)
        self.testset = torchvision.datasets.CIFAR10(root=root, train=False, transform=self._transform_test)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    def get_train_loader(self):
        return self.trainloader

    def get_test_loader(self):
        return self.testloader
