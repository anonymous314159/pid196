import torch
from .trainer import Trainer
from utils import AverageMeter, progress_bar


class AdversarialTrainer(Trainer):
    """
    An `AdversarialTrainer` object is used to handle all details in adversarial training process.
    """
    def __init__(self, model, criterion, optimizer, args):
        super().__init__(model, criterion, optimizer, args)
        self._epsilon = args.epsilon
        self._beta = args.beta

    def fgsm_attack(self, data: torch.Tensor):
        """
        Fast Gradient Sign Attack (FGSM)
        :return: perturbed_data
        """
        input_grad = data.grad.data
        perturbed_data = data + self._epsilon * input_grad
        return torch.clamp(perturbed_data, 0, 1)

    def _exec_single_epoch(self, loader, training, epoch):
        loss = AverageMeter()
        acc = AverageMeter()
        for i, (inputs, targets) in enumerate(loader):
            self._prepare_minibatch()
            targets = targets.to('cuda')
            inputs = inputs.to('cuda')
            inputs.requires_grad = True
            with torch.enable_grad():
                outputs = self._model(inputs)
                batch_loss = self._criterion(outputs, targets)
                # the gradient should be reserved to conduct FGSM attack.
                # But we don't update the parameters as normal `_backward()` does.
                self._backward(batch_loss, epoch, retain_graph=True, update=False)
            perturbed_inputs = self.fgsm_attack(inputs)
            perturbed_outputs = self._model(perturbed_inputs)
            prec1 = self.accuracy(perturbed_outputs.data, targets, topk=(1,))[0]
            batch_loss = batch_loss * self._beta + (1 - self._beta) * self._criterion(perturbed_outputs, targets)
            if training:
                # At this place, all parameters will be updated via back-propagation
                # to minimize the objective function in adversarial training.
                self._backward(batch_loss, epoch, retain_graph=False, update=True)
            loss.update(batch_loss.item(), inputs.size(0))
            acc.update(prec1.item(), inputs.size(0))
            progress_bar(i, len(loader), "Epoch={:3} | loss={:4.1f} | acc={:2.2f}".format(epoch, loss.val, acc.avg))
        return acc.avg
