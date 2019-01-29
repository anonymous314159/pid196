import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import WideResNet, resnet110, CifarResNeXt
from trainer import Trainer, OrthReguTrainer, AdversarialTrainer, AdversarialOrthReguTrainer
from dataset import CIFAR10, CIFAR100

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet',
                    help='name of CNN model. Choices: resnet, wideresnet, resnext')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 | cifar100')
parser.add_argument('--data', type=str, default='data', help='path of dataset')
parser.add_argument('-b', '--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', default=True, type=bool)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--ortho-decay', '--od', default=3e-2, type=float,
                    help='Weight decay of orthogonality regularization')
parser.add_argument('--adversarial', action='store_true', help='adversarial training')
parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in adversarial training')
parser.add_argument('--beta', type=float, default=0.5, help='beta in adversarial training')
parser.add_argument('--regu', type=str, default='no',
                    help='type of regularization. Possible values are: '
                         'no: no regularization'
                         'random-svd: employ random-svd in regularization ')


if __name__ == "__main__":
    args = parser.parse_args()
    # create model
    n_classes = args.dataset == 'cifar10' and 10 or 100
    if args.model == 'resnet':
        net = resnet110(num_classes=n_classes)
    elif args.model == 'wideresnet':
        net = WideResNet(depth=28, widen_factor=10, dropRate=0.3, num_classes=n_classes)
    elif args.model == 'resnext':
        net = CifarResNeXt(cardinality=8, depth=29, base_width=64, widen_factor=4, nlabels=n_classes)
    else:
        raise Exception('Invalid model name')
    # create optimizer
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    net.to('cuda')
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    # trainer
    if args.adversarial:
        if args.regu == 'no':
            trainer = AdversarialTrainer(net, criterion, optimizer, args)
        elif args.regu == 'random-svd':
            trainer = AdversarialOrthReguTrainer(net, criterion, optimizer, args)
        else:
            raise Exception('Invalid setting for adversarial training')
    else:
        if args.regu == 'no':
            trainer = Trainer(net, criterion, optimizer, args)
        elif args.regu == 'random-svd':
            trainer = OrthReguTrainer(net, criterion, optimizer, args)
        else:
            raise Exception('Invalid regularization term')
    # data
    if args.dataset == 'cifar100':
        data = CIFAR100(root=args.data, batch_size=args.batch_size)
    else:
        data = CIFAR10(root=args.data, batch_size=args.batch_size)
    # start
    best_acc = trainer.run(data, args.epochs)

