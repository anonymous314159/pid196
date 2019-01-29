from .orth_regu_trainer import OrthReguTrainer
from .adversarial_trainer import AdversarialTrainer


class AdversarialOrthReguTrainer(AdversarialTrainer, OrthReguTrainer):
    def __init__(self, model, criterion, optimizer, parser):
        super().__init__(model, criterion, optimizer, parser)

    def _backward(self, batch_loss, epoch, update=True, retain_graph=False):
        if retain_graph:
            return AdversarialTrainer._backward(self, batch_loss, epoch, update=update, retain_graph=True)
        else:
            return OrthReguTrainer._backward(self, batch_loss, epoch, update=update, retain_graph=False)
