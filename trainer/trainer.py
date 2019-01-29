import torch
import math
from utils import AverageMeter, progress_bar


class Trainer:
    """
    A `Trainer` object is used to handle the training process.
    """
    def __init__(self, model, criterion, optimizer, args):
        # prepare model
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._init_learning_rate = args.lr

    def _backward(self, batch_loss, epoch, update=True, retain_graph=False):
        self._optimizer.zero_grad()
        batch_loss.backward(retain_graph=retain_graph)
        if update:
            self._optimizer.step()

    def _prepare_minibatch(self):
        pass

    def adjust_learning_rate(self, epoch, num_epochs):
        # warmup
        warmup_period = 5
        if epoch < warmup_period:
            lr = self._init_learning_rate * epoch / warmup_period
        else:
            lr = 0.5 * (1 + math.cos((epoch - warmup_period) * math.pi / (num_epochs - warmup_period)))
            lr = lr * self._init_learning_rate
        # update learning rate
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _exec_single_epoch(self, loader, training, epoch):
        loss = AverageMeter()
        acc = AverageMeter()
        for i, (inputs, targets) in enumerate(loader):
            self._prepare_minibatch()
            targets = targets.to('cuda')
            inputs = inputs.to('cuda')
            outputs = self._model(inputs)
            batch_loss = self._criterion(outputs, targets)
            prec1 = self.accuracy(outputs.data, targets, topk=(1,))[0]
            loss.update(batch_loss.item(), inputs.size(0))
            acc.update(prec1.item(), inputs.size(0))
            if training:
                self._backward(batch_loss, epoch, retain_graph=False)
            #progress_bar(i, len(loader), "Epoch={:3} | loss={:4.1f} | acc={:2.2f}".format(epoch, loss.val, acc.avg))
            print('{}: {}'.format(i, acc.avg))
        return acc.avg

    def _train(self, epoch, loader):
        self._model.train()
        return self._exec_single_epoch(loader, True, epoch)

    def _test(self, epoch, loader):
        self._model.eval()
        acc = self._exec_single_epoch(loader, False, epoch)
        return acc

    def before_train(self, epoch, epochs):
        pass

    def run(self, data, epochs):
        try:
            for epoch in range(epochs + 1):
                self.adjust_learning_rate(epoch + 1, epochs)
                self.before_train(epoch, epochs)
                acc_train = self._train(epoch, data.get_train_loader())
                acc_test = self._test(epoch, data.get_test_loader())
        except Exception as err:
            torch.cuda.empty_cache()

    def accuracy(self, output, target, topk=(1,)):
        """
        Computes the precision@k for the specified values of k
        """
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

